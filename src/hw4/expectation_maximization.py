"""Mixtures of tree Bayesian networks using expectation-maximization (EM)."""

from functools import reduce
from operator import mul
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils.multiclass import unique_labels

from ._utils import bootstrap
from .tree import TreeBayesianNetworkClassifier


class ExpectationMaximizationTreeBayesianNetworkClassifier(
    BaseClassifier, ClassifierMixin
):
    def __init__(self, *, iterations: int = 100, bootstraps: int = 10):
        """Initializes this ``ExpectationMaximizationTreeBayesianNetworkClassifier``.

            Args:
                iterations: The maximum number of iterations of the EM algorithm to perform.
                bootstraps: The exact number of each of bootstrap samples and bagging models.
        """
        self.iterations = iterations
        self.bootstraps = bootstraps

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "ExpectationMaximizationTreeBayesianNetworkClassifier":
        """Fits this ``ExpectationMaximizationTreeBayesianNetworkClassifier``.

            Returns:
                ExpectationMaximizationTreeBayesianNetworkClassifier: This object, fitted.
        """
        # validate model parameters
        if self.iterations <= 0:
            raise ValueError(f"Iterations must be positive, but is {self.iterations}")
        if self.bootstraps <= 0:
            raise ValueError(f"Bootstraps must be positive, but is {self.bootstraps}")

        # validate method arguments
        if len(X) <= 0:
            raise ValueError(f"The length of X must be positive, but is {len(X)}")
        if len(X) != len(y):
            raise ValueError(
                f"The length of X and y must be equal, but are [{len(X)}, {len(y)}]"
            )

        # convenience variables
        k = self.bootstraps

        # initialize model
        clfs = [TreeBayesianNetworkClassifier().fit(*bootstrap(X, y)) for _ in range(k)]
        clfs_weights = pd.Series(np.full(k, 1 / k))

        # apply Expectation-Maximization (EM) algorithm
        is_converged = False
        i = 0
        while (i := i + 1) <= self.iterations and not is_converged:

            # "expect" likelihood of each observation for each mixture component
            expected = (clf.expect(X, y) for clf in clfs)

            weighted = (e * w for e, w in zip(expected, clfs_weights))
            df = pd.DataFrame({i_clf: sr for i_clf, sr in enumerate(weighted)})

            summed_x_row = df.apply(np.sum, axis=1)
            normalized_x_row = df.apply(lambda col: col / summed_x_row)  # DataFrame

            summed_x_col = normalized_x_row.sum()
            normalized_x_col = summed_x_col / summed_x_col.sum()  # Series

            # "maximize" mixture ensemble weights
            clfs_weights = normalized_x_col

            # "maximize" each mixture component
            row_weights_x_clf = (
                normalized_x_row.iloc[:, i_clf] / summed_x_col.iloc[i_clf]
                for i_clf in range(k)
            )
            clfs = [
                TreeBayesianNetworkClassifier().fit(X, y, row_weights)
                for row_weights in row_weights_x_clf
            ]

            # TODO test for convergence (to be able to stop early)

        self.classifiers_ = clfs
        self.weights_ = clfs_weights
        return self
