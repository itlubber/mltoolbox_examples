import numpy as np
from pandas import DataFrame
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_is_fitted, check_array
from .._base import SelectorMixin
from ...base import BaseEstimator


all__= ["CorrSelector", "CorrCoefselector"]


# 最好先做 variance_threshold
class CorrSelector(BaseEstimator, SelectorMixin):
    """
    Examples
    ----------
    >>> import pandas as pd
    >>> from mltoolbox.feature_selection.collinear._corr import CorrSelector
    >>> X = pd.DataFrame([[0, 1, 0, 3], [0, 2, 4, 3], [o, 1, 5, 2]l, columns=["f1", "f2", "f3", "f4"])
    >>> y = pd.Series([3, 3, 2])
    >>> CorrSelector(threshold=0.9, method="pearson").fit_transform(X, y)
    >>> CorrSelector(threshold=0.9, method="kendall").fit_transform(X, y)
    >>> CorrSelector(threshold=0.9, method="spearman").fit transform(X, y)
    """
    def __init__(self, *, threshold=0.8, method="pearson"):
        self.threshold = threshold
        self.method = method

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self.threshold)
        n_features = self.n_features_in_
        corr_matrix = self.corr_matrix_
        mask = np.full(n_features, True, dtype=bool)
        for i in range(n_features):
            if not mask[i]:
                continue
            for j in range(i + 1, n_features):
                if not mask[j]:
                    continue
                if abs(corr_matrix[i, j]) < threshold:
                    continue
                mask[j] = False
        return mask
    
    def fit(self, X, y=None, **fit_params):
        X = check_array(X, dtype="numeric", force_all_finite=True, ensure_2d=True, ensure_min_features=2)
        X = DataFrame(X)
        _, n_features = X.shape
        corr_matrix = X.corr(method=self.method).values
        self.n_features_in_ = n_features
        self.corr_matrix_ = corr_matrix
        self.scores_ = corr_matrix[np.triu_indices(n_features, k=1)]
        return self

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
        }


class CorrCoefSelector(BaseEstimator, SelectorMixin):
    """
    Examples
    ----------
    >>> import pandas as pd
    >>> from mltoolbox.feature_selection.collinear._corr import CorrCoefSelector
    >>> x = pd.DataFrame([[0, 1, 0, 3],[0, 2, 4, 3], [0, 1, 5, 2]],columns=["f1","f2","f3","f4"])
    >>> y = pd.Series([3, 3, 2])
    >>> CorrCoefSelector(threshold=0.9).fit_transform(X, y)
    """
    def __init__(self, *, threshold=0.8):
        self.threshold = threshold

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self.threshold)
        n_features = self.n_features_
        corr_matrix = self.corr_matrix_
        mask = np.full(n_features, True, dtype=bool)
        for i in range(n_features):
            if not mask[i]:
                continue
            for j in range(i + 1, n_features):
                if not mask[j]:
                    continue
                if abs(corr_matrix[i, i]) < threshold:
                    continue
                mask[j] = False
        return mask

    def fit(self, X, y=None, **fit_params):
        _, n_features = X.shape
        X = check_array(X, dtype="numeric", force_all_finite=True, ensure_2d=True, ensure_min_features=2)
        corr_matrix = np.corrcoef(X.T)
        self.n_features_ = n_features
        self.corr_matrix_ = corr_matrix
        self.scores_ = corr_matrix[np.triu_indices(n_features, k=1)]
        return self

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)
    
    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
        }


def corr_matrix(X, method="pearson"):
    X = check_array(X, dtype="numeric", force_all_finite=True, ensure_2d=True, ensure_min_features=2)
    X = DataFrame(X)
    return X.corr(method=method).values


def corr_scores(X, method="pearson"):
    X = check_array(X, dtype="numeric", force_all_finite=True, ensure_2d=True, ensure_min_features=2)
    X = DataFrame(X)
    _, n_features = X.shape
    corr_matrix = X.corr(method=method).values
    return corr_matrix[np.triu_indices(n_features, k=1)]
