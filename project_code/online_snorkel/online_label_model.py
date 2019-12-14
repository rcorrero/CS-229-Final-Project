import logging
import pickle
import random
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from munkres import Munkres  # type: ignore

from snorkel.analysis import Scorer
from snorkel.labeling.analysis import LFAnalysis
from snorkel.labeling.model.graph_utils import get_clique_tree
from snorkel.labeling.model.logger import Logger
from snorkel.labeling import LabelModel
from snorkel.labeling.model.label_model import TrainConfig
from snorkel.types import Config
from snorkel.utils import probs_to_preds
from snorkel.utils.config_utils import merge_config
from snorkel.utils.lr_schedulers import LRSchedulerConfig
from snorkel.utils.optimizers import OptimizerConfig

from r_pca import R_pca


class OnlineLabelModel(LabelModel):
    def __init__(self, cardinality: int = 2, **kwargs: Any) -> None:
        super().__init__(cardinality, **kwargs)
        self.is_trained = False


    def fit(
        self,
        L_train: np.ndarray,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        threshold: Optional[float] = 1e-16,
        **kwargs: Any,
    ) -> None:
        self.edges = []
        self._set_constants(L_train + 1)  # L_train + 1 == L_shift
        self._create_moments(L_train + 1) # L_train + 1 == L_shift
        self.threshold = threshold
        super().fit(L_train,
                    Y_dev,
                    class_balance,
                    **kwargs
        )
        self.is_trained = True


    def _update_O(self, L: np.ndarray, alpha: float, higher_order: bool = False) -> None:
        """Update overlaps and conflicts matrix from label matrix.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}
        alpha
            Exponential smoothing factor
        higher_order
            Whether to include higher-order correlations (e.g. LF pairs) in matrix
        """            
        L_aug = self._get_augmented_label_matrix(L, higher_order=higher_order)
        self.d = L_aug.shape[1]
        update = L_aug.T @ L_aug / self.n
        O = self.O.cpu().detach().numpy()

        self.O = (
            torch.from_numpy(
                (1 - alpha) * O + alpha * update
            ).float().to(self.config.device)
        )

    
    def _update_balance(self,
                          class_balance: Optional[List[float]] = None,
                          Y_dev: Optional[np.ndarray] = None,
    ) -> None:
        # TODO: Implement this method
        raise NotImplementedError('`_update_balance()` not implemented.')


    def _robust_pca(self, sigma: np.ndarray, **kwargs: Any) -> None:
        r_pca = R_pca(sigma, **kwargs)
        L, S = r_pca.fit()
        return L, S


    def _create_moments(self, L_shift: np.ndarray) -> None:
        # Calculate \hat{M}
        M = L_shift.T @ L_shift / self.n
        self.M = M

        # Calculate \hat{\mu}
        means = np.mean(L_shift, axis = 0)
        means.shape = (self.m, 1)
        self.means = means

        # Calculate \hat{\Sigma}
        sigma = self.M - self.means @ self.means.T # Means must have shape (m,1)
        self.sigma = sigma

        # _robust_pca(\hat{\Sigma})
        L, S = self._robust_pca(self.sigma)
        self.L = -L # L = -z @ z.T
        self.S = S

    def _create_tree(self) -> None:
        edges = []
        for row in range(self.m):
            for col in range(row + 1, self.m):
                if self.S[row,col] > self.threshold:
                    edges.append((row,col))
        if edges != self.edges or not self.is_trained:
            self.c_tree = get_clique_tree(range(self.m), edges)
            self.edges = edges
        
 
    def _update_tree(self, L_shift: np.ndarray, alpha: float, threshold: float) -> None:
        # Calculate \hat{M}
        M_update = L_shift.T @ L_shift / self.n
        M = self.M
        self.M = (1 - alpha) * M + alpha * M_update

        # Calculate \hat{\mu}
        means_update = np.mean(L_shift, axis = 0)
        means_update.shape = (self.m,1)
        means = self.means
        self.means = (1 - alpha) * means + alpha * means_update

        # Calculate \hat{\Sigma}
        self.sigma = self.M - self.means @ self.means.T # Means must have shape (m,1)

        # _robust_pca(\hat{\Sigma})
        L, S = self._robust_pca(self.sigma)
        self.L = -L # L = -z @ z.T
        self.S = S

        self._create_tree()

    
    def _update_mask(self) -> None:
        self._build_mask()    
        

    def partial_fit(
        self,
        L_train: np.ndarray,
        alpha: Optional[float] = 0.05,
        Y_dev: Optional[np.ndarray] = None,
        class_balance: Optional[List[float]] = None,
        update_balance: bool = False,
        update_tree: bool = False,
        threshold: Optional[float] = 1e-16,
        **kwargs: Any,
    ) -> None:
        """Train label model.
        Train label model to estimate mu, the parameters used to combine LFs.
        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        alpha
            Exponential smoothing factor, by default 0.05
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
        class_balance
            Each class's percentage of the population, by default None
        **kwargs
            Arguments for changing train config defaults
        Raises
        ------
        Exception
            If loss in NaN
        Examples
        --------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> Y_dev = [0, 1, 0]
        >>> label_model = LabelModel(verbose=False)
        >>> label_model.fit(L)
        >>> label_model.fit(L, Y_dev=Y_dev)
        >>> label_model.fit(L, class_balance=[0.7, 0.3])
        """
        if not self.is_trained:
            raise RuntimeError(f"This instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")
        n, m = L_train.shape
        if m != self.m:
            raise ValueError(f"L_train must have shape[1]={self.m}.")
            
        # Set number of epochs to one
        self.train_config: TrainConfig = merge_config(  # type:ignore
            TrainConfig(), {"n_epochs": 1, **kwargs}  # type:ignore
        )

        L_shift = L_train + 1  # convert to {0, 1, ..., k}
        if L_shift.max() > self.cardinality:
            raise ValueError(
                f"L_train has cardinality {L_shift.max()}, cardinality={self.cardinality} passed in."
            )
        
        self._set_constants(L_shift)
        if update_balance:
            self._update_balance(class_balance, Y_dev)
        if update_tree:
            self._update_tree(L_train, alpha, threshold)
            # Build the mask over O^{-1}
            self._update_mask()
            
        lf_analysis = LFAnalysis(L_train)
        self.coverage = lf_analysis.lf_coverages()

        # Compute O
        if self.config.verbose:  # pragma: no cover
            logging.info("Computing O...")
        
        self._update_O(L_shift, alpha)

        # Estimate \mu
        if self.config.verbose:  # pragma: no cover
            logging.info("Estimating \mu...")

        # Set model to train mode
        self.train()

        # Move model to GPU
        if self.config.verbose and self.config.device != "cpu":  # pragma: no cover
            logging.info("Using GPU...")
        self.to(self.config.device)

        # Set training components
        self._set_optimizer()

        # Restore model
        start_iteration = 0

        # Train the model
        metrics_hist = {}  # The most recently seen value for all metrics
        for epoch in range(start_iteration, self.train_config.n_epochs):
            self.running_loss = 0.0
            self.running_examples = 0

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass to calculate the loss
            loss = self._loss_mu(l2=self.train_config.l2)
            if torch.isnan(loss):
                msg = "Loss is NaN. Consider reducing learning rate."
                raise Exception(msg)

            # Backward pass to calculate gradients
            # Loss is an average loss per example
            loss.backward()

            # Perform optimizer step
            self.optimizer.step()

            # Calculate metrics, log, and checkpoint as necessary
            metrics_dict = self._execute_logging(loss)
            metrics_hist.update(metrics_dict)

            # Update learning rate
            self._update_lr_scheduler(epoch)

        # Post-processing operations on mu
        self._clamp_params()
        self._break_col_permutation_symmetry()

        # Return model to eval mode
        self.eval()

        # Print confusion matrix if applicable
        if self.config.verbose:  # pragma: no cover
            logging.info("Finished Training")

        self.is_trained = True
