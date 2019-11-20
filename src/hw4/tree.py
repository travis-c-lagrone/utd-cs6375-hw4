"""Tree Bayesian networks."""

import networkx as nx
import numpy as np
import pandas as pd

from functools import reduce
from itertools import product
from operators import mul
from typing import Dict, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from sklearn.utils.multiclass import unique_labels

# from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class TreeBayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    classes_: np.ndarray
    features_: np.ndarray
    tree_: nx.Graph

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeBayesianNetworkClassifier":
        if len(X) != len(y):
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: [{len(X)}, {len(y)}]"
            )

        self.classes_ = unique_labels(y)
        self.features_ = np.array(X.columns)

        G = nx.Graph()
        N = len(y)

        # add nodes
        for col in X.columns:
            sr = X[col]
            probs = sr.groupby(sr).count() / N  # relative frequencies
            n_dims = len(probs)  # arity of the ``col`` domain
            G.add_node(col, probs=probs, n_dims=n_dims)

        # add edges
        for i_f1 in range(len(X.columns) - 1):
            for i_f2 in range(i_f1 + 1, len(X.columns)):
                cols = sorted([X.columns[i_f1], X.columns[i_f2]])
                contingency = contingency_matrix(*X[cols])
                n_dims = reduce(mul, contingency.shape)  # arity of the ``*cols`` domain
                probs = (contingency + 1) / (N + n_dims)  # uses 1-Laplace smoothing
                mutual_info = mutual_info_score(None, None, contingency)
                G.add_edge(*cols, probs=probs, n_dims=n_dims, weight=mutual_info)

        # extract maximum spanning tree by mutual information
        self.tree_ = nx.maximum_spanning_tree(G)

        return self
