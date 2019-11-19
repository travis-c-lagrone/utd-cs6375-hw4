"""Independent Bayesian networks."""

from collections import defaultdict
from typing import Dict, List

from numpy.core import ndarray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class IndependentBayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    classes_: List[int]
    labels_by_feature_: Dict[int, List[int]]
    probability_of_class_: Dict[int, float]
    probability_of_label_by_feature_given_class_: Dict[int, Dict[int, Dict[int, float]]]

    def fit(self, X: ndarray, y: ndarray) -> "IndependentBayesianNetworkClassifier":
        X_, y_ = check_X_y(X, y)
        n, d = X_.shape

        self.classes_ = list(unique_labels(y_))
        self.labels_by_feature_ = {f: unique_labels(X[:, f]) for f in range(d)}
        self.probability_of_class_ = {}
        self.probability_of_label_by_feature_given_class_ = {}

        for c in self.classes_:
            mask_c = y == c
            n_c = sum(mask_c)
            X_c = X_[mask_c]
            y_c = y_[mask_c]

            prob_of_label_of_feature: Dict[int, Dict[int, float]] = {}
            for f in range(d):
                X_c_f = X_c[:, f]

                nd_f = len(self.labels_by_feature_[f])
                n_c_smoothed = n_c + nd_f  # uses 1-Lapace smoothing
                default_prob = 1 / n_c_smoothed  # uses 1-Lapace smoothing
                factory = lambda dp=default_prob: dp
                prob_of_label: Dict[int, float] = defaultdict(factory)

                for l in self.labels_by_feature_[f]:
                    mask_c_f_v = X_c_f == l
                    n_c_f_v = sum(mask_c_f_v)
                    prob = (n_c_f_v + 1) / n_c_smoothed  # uses 1-Lapace smoothing
                    prob_of_label[l] = prob

                prob_of_label_of_feature[f] = prob_of_label

            # fmt: off
            self.probability_of_class_[c] = n_c / n
            self.probability_of_label_by_feature_given_class_[c] = prob_of_label_of_feature
            # fmt: on

        return self

    def predict(self, X: ndarray) -> ndarray:
        check_is_fitted(self, ["X_", "y_"])
        X_ = check_array(X)
        raise NotImplementedError()  # TODO
