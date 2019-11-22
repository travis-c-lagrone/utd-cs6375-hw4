"""Tree Bayesian networks."""

from functools import reduce
from itertools import product
from operator import mul
from typing import ClassVar, Dict, Final, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.utils.multiclass import unique_labels

from ._utils import root_tree


class TreeBayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    classes_: np.ndarray
    features_: np.ndarray
    network_: nx.DiGraph

    _X_COL_PREFIX = "X_"
    _Y_COL_PREFIX = "y_"

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "TreeBayesianNetworkClassifier":
        if len(X) != len(y):
            raise ValueError(
                "Found input variables with inconsistent number of samples: "
                f"[{len(X)}, {len(y)}]"
            )

        self.classes_ = unique_labels(y)
        self.features_ = np.array(X.columns)

        data = X.add_prefix(__class__._X_COL_PREFIX)
        data[__class__._Y_COL_PREFIX] = y

        G = nx.Graph()
        N = len(data)

        # add nodes
        for col in data.columns:
            sr = data[col]
            probs = sr.groupby(sr).count() / N  # groupby returns with presorted index
            labels = unique_labels(sr)
            G.add_node(col, probs=probs, labels=labels)

        # add edges
        for i_f1 in range(len(data.columns) - 1):
            for i_f2 in range(i_f1 + 1, len(data.columns)):
                cols = sorted([data.columns[i_f1], data.columns[i_f2]])
                contingency = contingency_matrix(*data[cols])
                mutual_info = mutual_info_score(None, None, contingency)

                # compute joint probability distribution
                nd = reduce(mul, contingency.shape)  # arity of the ``*cols`` domain
                probs = (contingency + 1) / (N + nd)  # uses 1-Laplace smoothing
                df = pd.DataFrame(probs)
                df.index = G.nodes[cols[0]]["labels"]
                df.columns = G.nodes[cols[-1]]["labels"]
                sr = df.stack()
                sr.index.names = cols

                G.add_edge(*cols, joint_probs=sr, mutual_info=mutual_info)

        # extract maximum spanning tree by mutual information
        T = nx.maximum_spanning_tree(G, weight="mutual_info")
        arborescence = root_tree(T, root=__class__._Y_COL_PREFIX)
        self.network_ = arborescence

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:  # TODO
        if set(self.features_) != set(X.columns):
            raise ValueError(
                f"Feature sets do not match: [{self.features_}, {X.columns}]"
            )

        data = X.add_prefix(__class__._X_COL_PREFIX)
