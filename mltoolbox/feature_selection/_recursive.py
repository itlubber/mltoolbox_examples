from abc import abstractmethod
import numpy as np
from pandas import DataFrame
from sklearn import clone
from sklearn.base import MetaEstimatorMixin, is_classifier, is_regressor
from sklearn.feature_selection import RFE
from sklearn.utils import check_array
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args

from ..base import BaseEstimator, TransformerMixin


class BaseSorter(BaseEstimator, TransformerMixin):
    _invert = False

    @abstractmethod
    def _get_feature_scores(self):
        pass

    def get_sorted_features(self, X):
        scores = self._get_feature_scores()
        if self._invert:
            indices = np.argsort(scores, kind='mergesort')
        else:
            indices = np.argsort(-scores, kind='mergesort')
        # scores[indices] is monotonic decreasing
        if isinstance(X, DataFrame):
            columns = X.columns.values
        else:
            columns = np.arange(X.shape[1])
        # only sort the columns
        return columns[indices]
    
    def transform(self, X):
        Xt = check_array(X, dtype=None, accept_sparse=False, force_all_finite=not _safe_tags(self, key="allow_nan"))
        _, n_features = Xt.shape
        scores = self._get_feature_scores()
        if len(scores) != n_features:
            raise ValueError("X has a different shape than during fitting.")
        if self._invert:
            indices = np.argsort(scores, kind='mergesort')
        else:
            indices = np.argsort(-scores, kind='mergesort')
        # scores[indices] is monotonic decreasing
        if isinstance(X, DataFrame):
            columns = X.columns.values
        else:
            columns = np.arange(X.shape[1])
        # only sort the columns
        ranked_features = columns[indices]
        if isinstance(X, DataFrame):
            return X[ranked_features]
        return X[:, ranked_features]


class BaseRecursiveSorter(BaseSorter):
    _invert = True

    def __init__(self, *, n_features_to_select=None, step=1, verbose=0, importance_getter="auto"):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.importance_getter = importance_getter

    def _make_estimator(self):
        self.estimator = None

    def fit(self, X, y=None):
        self._make_estimator()
        rfe = RFE(estimator=clone(self.estimator), n_features_to_select=self.n_features_to_select, step=self.step,
        verbose=self.verbose, importance_getter=self.importance_getter)
        rfe.fit(X, y)
        self.rfe = rfe
        self.scores_ = rfe.ranking_
        return self
    
    def _get_feature_scores(self):
        check_is_fitted(self, "scores_")
        return self.scores_


class RecursiveSorter(BaseRecursiveSorter, MetaEstimatorMixin):
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_features_to_select=None, step=1, verbose=0):
        BaseRecursiveSorter.__init__(self, n_features_to_select=n_features_to_select, step=step, verbose=verbose)
        self.estimator = estimator

    def _make_estimator(self):
        if self.estimator is None:
            raise ValueError("Estimator should not be None!")
        if not (is_classifier(self.estimator) or is_regressor(self.estimator)):
            raise ValueError("For recursive ranked selector, estimator should be a classifier or regressor.")
