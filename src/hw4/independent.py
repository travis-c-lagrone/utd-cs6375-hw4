"""Independent Bayesian networks."""

from typing import Dict, List

from numpy.core import ndarray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class IndependentBayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    classes_: List[int]
    probability_of_class_: Dict[int, float]
    probability_of_value_of_feature_given_class_: Dict[int, Dict[int, Dict[int, float]]]

    def fit(self, X: ndarray, y: ndarray) -> "IndependentBayesianNetworkClassifier":
        X_, y_ = check_X_y(X, y)
        n, d = X_.shape

        self.classes_ = list(unique_labels(y_))
        self.probability_of_class_ = {}
        self.probability_of_value_of_feature_given_class_ = {}

        for c in self.classes_:  # type: int
            mask_c = y == c
            n_c = sum(mask_c)
            X_c = X_[mask_c]
            y_c = y_[mask_c]

            probability_of_value_of_feature: Dict[int, Dict[int, float]] = {}
            for f in range(d):  # type: int
                X_c_f = X_c[:, f]

                probability_of_value: Dict[int, float] = {}
                for v in unique_labels(X_c_f):  # type: int
                    mask_c_f_v = X_c_f == v
                    n_c_f_v = sum(mask_c_f_v)
                    probability = n_c_f_v / n_c

                    probability_of_value[v] = probability

                probability_of_value_of_feature[f] = probability_of_value

            # fmt: off
            self.probability_of_class_[c] = n_c / n
            self.probability_of_value_of_feature_given_class_[c] = probability_of_value_of_feature
            # fmt: on

        return self

    def predict(self, X: ndarray) -> ndarray:
        check_is_fitted(self, ["X_", "y_"])
        X_ = check_array(X)
        raise NotImplementedError()  # TODO
