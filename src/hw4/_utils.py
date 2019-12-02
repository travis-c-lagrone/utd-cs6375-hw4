import math as m

from typing import Any, Dict, TypeVar

import networkx as nx
import numpy as np
import pandas as pd

from networkx.algorithms.tree.recognition import is_tree
from scipy import sparse

# fmt: off


def _create_dict_typevars():
    return TypeVar("K"), TypeVar("V")


def iter_rows(nd: np.ndarray) -> np.ndarray:
    """Iterates through the rows of a 2d array."""
    for i in range(nd.shape[0]):
        yield nd[i, :]


def iter_cols(nd: np.ndarray) -> np.ndarray:
    """Iterates through the columns of a 2d array."""
    for j in range(nd.shape[1]):
        yield nd[:, j]


K, V = _create_dict_typevars()
def max_key_by_val(d: Dict[K, V]) -> K:
    return max(d.items(), key=lambda kv: kv[1])[0]  # type: ignore


def smoothed_estimate(n_x: int, N: int, d: int) -> float:
    """Estimates with 1-Laplace smoothing the probability of a category from a multinomial distribution.

        Args:
            n_x (int): The count of some outcome "x" among ``N`` trials.
                SHOULD be non-negative.
                SHOULD be no greater than ``N``.
            N (int): The count of trials.
                SHOULD be non-negative.
            d (int): The count of distinct possible outcomes.
                (i.e. the dimensionality of the distribution)
                SHOULD be positive.

        Returns:
            float: The estimated probability with 1-Laplace smoothing of some outcome "x".
    """
    return (n_x + 1) / (N + d)


def root_tree(tree: nx.Graph, root: Any) -> nx.DiGraph:
    r"""Root an undirected tree.

        Args:
            tree (nx.Graph): The undirected tree to root.
            root (Any): The node in ``tree`` to use as the root.

        Raises:
            ValueError: If ``tree`` is not a tree.
            ValueError: If ``root`` is not in ``tree`.

        Returns:
            nx.DiGraph: The rooting of ``tree`` at ``root``.
    """
    if not is_tree(tree):
        raise ValueError("The graph is not a tree")
    if root not in tree:
        raise ValueError(f"The root {root} is not in the tree")

    arborescence = nx.DiGraph(tree.nodes)
    S = {tree.nodes[root]}
    Q = [tree.nodes[root]]
    for u, v, d in tree.nodes[Q.pop()].edges(data=True):
        arborescence.add_edge(u, v, **d)
        S.add(v)
        if v not in S:
            Q.append(v)

    return arborescence


def bootstrap(X: pd.DataFrame) -> pd.DataFrame:
    return X.iloc[np.random.randint(N := len(X), size=N)]


def weighted_contingency_matrix(
    labels_true: pd.Series, labels_pred: pd.Series, weights: pd.Series
) -> np.ndarray:
    df = pd.DataFrame({"labels_true": labels_true, "labels_pred": labels_true, "weights": weights})
    gb = df.groupby(["labels_true", "labels_pred"])[["weights"]].sum()
    pt = pd.pivot_table(gb, index="labels_true", columns="labels_pred", aggfunc="sum", fill_value=0)
    return pt.values


def weighted_mutual_info_score(weighted_contingency: np.ndarray) -> float:
    """A substitute for ``sklearn.metrics.mutual_info_score`` that accommodates floating-point values.

        The implementation of this method is almost entirely copy-pasted from Scikit-Learn v0.21.3.
    
    """
    if isinstance(weighted_contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(weighted_contingency)
        nz_val = weighted_contingency[nzx, nzy]
    elif sparse.issparse(weighted_contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sparse.find(weighted_contingency)
    else:
        raise ValueError(
            f"Unsupported type for 'weighted_contingency': {type(weighted_contingency)}"
        )

    contingency_sum = weighted_contingency.sum()
    pi = np.ravel(weighted_contingency.sum(axis=1))
    pj = np.ravel(weighted_contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = (pi.take(nzx).astype(np.float64, copy=False)
           * pj.take(nzy).astype(np.float64, copy=False))
    log_outer = -np.log(outer) + m.log(pi.sum()) + m.log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - m.log(contingency_sum))
        + contingency_nm * log_outer)
    return mi.sum()


__import__ = __dir__ = sorted([
    "iter_rows",
    "iter_cols",
    "max_key_by_val",
    "smoothed_estimate",
    "root_tree",
    "bootstrap",
    "weighted_contingency_matrix",
    "weighted_mutual_info_score",
])
