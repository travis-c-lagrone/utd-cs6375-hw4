from numpy import ndarray
from typing import Dict, TypeVar


def iter_rows(nd: ndarray) -> ndarray:
    for i in range(nd.shape[0]):
        yield nd[i, :]


def iter_cols(nd: ndarray) -> ndarray:
    for j in range(nd.shape[1]):
        yield nd[:, j]


K = TypeVar("K")
V = TypeVar("V")


def max_key_by_val(d: Dict[K, V]) -> K:
    return max(d.items(), key=lambda kv: kv[1])[0]
