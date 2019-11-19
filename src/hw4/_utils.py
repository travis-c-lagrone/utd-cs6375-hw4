from numpy import ndarray
from typing import Dict, TypeVar

# fmt: off


def _create_dict_typevars():
    return TypeVar("K"), TypeVar("V")


def iter_rows(nd: ndarray) -> ndarray:
    """Iterates through the rows of a 2d array."""
    for i in range(nd.shape[0]):
        yield nd[i, :]


def iter_cols(nd: ndarray) -> ndarray:
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


__import__ = __dir__ = sorted([
    "iter_rows",
    "iter_cols",
    "max_key_by_val",
])
