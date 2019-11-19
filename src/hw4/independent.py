"""Independent Bayesian networks."""

from collections import defaultdict
from functools import reduce
from math import log
from operator import mul
from typing import Dict, Tuple

from numpy import array
from numpy.core import ndarray
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._utils import iter_rows, max_key_by_val


class IndependentBayesianNetworkClassifier(BaseEstimator, ClassifierMixin):
    classes_: Tuple[int]
    features_: Tuple[int]
    labels_by_feature_: Dict[int, Tuple[int]]
    probability_of_class_: Dict[int, float]
    probability_of_label_by_feature_given_class_: Dict[int, Dict[int, Dict[int, float]]]

    def fit(self, X: ndarray, y: ndarray) -> "IndependentBayesianNetworkClassifier":
        X_, y_ = check_X_y(X, y)
        n, d = X_.shape

        self.classes_ = tuple(unique_labels(y_))
        self.features_ = tuple(range(d))
        self.labels_by_feature_ = {f: tuple(unique_labels(X_[:, f])) for f in range(d)}
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
        # fmt: off
        check_is_fitted(self, [
            "classes_",
            "features_",
            "labels_by_feature_",
            "probability_of_class_",
            "probability_of_value_of_feature_given_class_",
        ])
        X_ = check_array(X)
        # fmt: on

        preds: List[int] = []
        for row in iter_rows(X_):

            logprob_by_c: Dict[int, float] = []
            for c in self.classes_:

                factors = [self.probability_of_class_[c]]
                for f in self.features_:
                    l = row[f]
                    factor = self._get_prob(c, f, l)
                    factors.append(factor)

                logs = map(log, factors)
                logprob = reduce(mul, logs)
                logprob_by_c[c] = logprob

            max_c = max_key_by_val(logprob_by_c)
            preds.append(max_c)

        return array(preds)

    def _get_prob(self, class_: int, feature: int, label: int) -> float:
        return self.probability_of_label_by_feature_given_class_[class_][feature][label]
