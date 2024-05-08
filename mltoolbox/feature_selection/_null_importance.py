import numbers
from abc import abstractmethod
import numpy as np
from sklearn.feature_selection._from_model import _calculate_threshold, _get_feature_importances
from sklearn.model_selection._split import check_cv
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args

from ._base import SelectorMixin
from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier


class BaseNullImportanceSelection(BaseEstimator, SelectorMixin):
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1, top_k=None, importance_getter='auto', cv=3, n_runs=5):
        self.threshold = threshold
        self.norm_order = norm_order
        self.top_k = top_k
        self.importance_getter = importance_getter
        self.cv = cv
        self.n_runs = n_runs

    @abstractmethod
    def _make_estimator(self):
        self.estimator = None

    def fit(self, X, y=None):
        self._make_estimator()
        if self.top_k is not None:
            if not isinstance(self.top_k, numbers.Integral):
                raise TypeError("top_k should be an integer between 0 and {} features. Got {!r} instead.".format(X.shape[1], self.top_k))
            
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        random_state = cv.random_state
        np.random.seed(random_state)

        X, y = self._validate_data(X, y)

        n_splits = cv.get_n_splits()
        n_runs = self.n_runs
        getter = self.importance_getter
        norm_order = self.norm_order

        estimator = clone(self.estimator)
        #计算shuffle之后的特征重要性
        n_samples, n_features = X.shape
        null_importances = np.zeros((n_features, n_splits * n_runs))
        idx = np.arange(n_samples)
        for run in range(n_runs):
            np.random.shuffle(idx)
            y_shuffled = y[idx]

            for fold_, (train_idx, valid_idx) in enumerate(cv.split(y_shuffled, y_shuffled)):
                estimator.fit(X[train_idx], y_shuffled[train_idx].ravel())
                null_importance = _get_feature_importances(estimator, getter, transform_func=None, norm_order=norm_order)
                null_importances[:, n_splits * run + fold_] = null_importance
            
        estimator = clone(self.estimator)
        # 计算未shuffle的特征重要性
        actual_importances = np.zeros((n_features, n_splits * n_runs))
        for run in range(n_runs):
            for fold_, (train_idx, valid_idx) in enumerate(cv.split(y, y)):
                estimator.fit(X[train_idx], y[train_idx].ravel())
                actual_importance = _get_feature_importances(estimator, getter, transform_func=None, norm_order=norm_order)
                actual_importances[:, n_splits * run + fold_] = actual_importance

        self.null_importances = null_importances
        self.actual_importances_ = actual_importances

        scores = np.zeros(n_features)
        for i in range(n_features):
            scores[i] = _feature_score_v2(actual_importances[i, :], null_importances[i, :])
        self.scores_ = scores
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, ["scores_"])
        scores = self.scores_
        threshold = _calculate_threshold(self.estimator, scores, self.threshold)
        if self.top_k is not None:
            mask = np.zeros_like(scores, dtype=bool)
            candidate_indices = np.argsort(-scores, kind='mergesort')[:self.top_k]
            mask[candidate_indices] = True
        else:
            mask = np.ones_like(scores, dtype=bool)
        mask[scores < threshold] = False
        return mask


class NullImportanceSelection(BaseNullImportanceSelection, MetaEstimatorMixin):
    @_deprecate_positional_args
    def __init__(self, estimator, *, threshold='mean', norm_order=1, top_k=None, cv=3, n_runs=5):
        BaseNullImportanceSelection.__init__(self, threshold=threshold, norm_order=norm_order, top_k=top_k, cv=cv, n_runs=n_runs)
        self.estimator = estimator
    
    def _make_estimator(self):
        if self.estimator is None:
            raise ValueError("Estimator should not be None!")


def _feature_score_v1(actual_importances, null_importances):
    #未进行特征shuffle的特征重要性除以shuffle以后的0.75分位数作为score
    actual_importance = actual_importances.mean()
    return np.log(1e-10 + actual_importance / (1. + np.percentile(null_importances, 75)))


def _feature_score_v2(actual_importances, null_importances):
    # shuffle之后特征重要性低于实际target对应特征的重要性0.25分位数的次数百分比
    return 100 * (np.count_nonzero(null_importances < np.percentile(actual_importances, 25)) / null_importances.shape[0])
