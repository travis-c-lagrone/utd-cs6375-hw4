"""Tree Bayesian networks."""

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

from ._utils import root_tree


class TreeBayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    classes_: np.ndarray
    features_: np.ndarray
    network_: nx.DiGraph

    _X_COL_PREFIX = "X_"
    _Y_COL_PREFIX = "y_"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeBayesianNetworkClassifier":
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

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if set(self.features_) != set(X.columns):
            raise ValueError(
                f"Feature sets do not match: [{self.features_}, {X.columns}]"
            )

        data = X.add_prefix(__class__._X_COL_PREFIX)
        preds = map(self._predict_row, data.iterrows())
        return pd.Series(preds, index=data.index)

    def _predict_row(self, row: pd.Series) -> Any:
        return self._compute_post_probs(row).idxmax()

    def _compute_post_probs(
        self, row: pd.Series, node=__class__._Y_COL_PREFIX
    ) -> pd.Series:
        children_probs = []
        for child in self.network_.successors(node):
            child_val = row[child]
            joint_probs = self.network_.edges[(node, child)]["joint_probs"]
            if child_val is not None:  # base case 1/2
                if child < node:
                    sliced = joint_probs[child_val]
                else:
                    sliced = joint_probs[:, child_val]
                child_probs = sliced / sliced.sum()  # normalize
            elif not list(self.network_.successors(child)):  # base case 2/2
                child_probs = joint_probs.groupby(child).sum()  # sum-out
            else:  # recursive case
                recursed = self._compute_post_probs(row, child)  # univariate
                merged = pd.merge(recursed, joint_probs, on=child)
                multiplied = pd.Series(
                    reduce(mul, [merged[col] for col in merged.columns])
                )
                summed = multiplied.groupby(child).sum()  # sum-out
                child_probs = summed / summed.sum()
                raise NotImplementedError(
                    "Recursive case of compute_post_probs(node)"
                    " within TreeBayesianNetworkClassifier.predict(self, X)"
                )
            children_probs.append(child_probs)
        reduced = reduce(mul, children_probs)  # NOTE assumes same order
        normalized = reduced / reduced.sum()
        return normalized
