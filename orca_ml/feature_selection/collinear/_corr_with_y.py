import numpy as np
from pandas import DataFrame, Series
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_is_fitted, check_X_y, _deprecate_positional_args
from ...base import BaseEstimator
from .._base import SelectorMixin


# not suggest to handle features with low variance
class PearsonWithYSelector(BaseEstimator, SelectorMixin):
    @_deprecate_positional_args
    def __init__(self, *, threshold=0.7):
        self.threshold = threshold
    
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        threshold = _calculate_threshold(self, scores, self.threshold)
        return np.abs(scores) > threshold
    
    def fit(self, X, y=None, **fit_params):
        scores = corr_with_y(X, y, method="pearson")
        self.scores_ = scores
        return self

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        return _calculate_threshold(self, scores, self.threshold)

    def _more_tags(self):
        return {
            "X_types": ["2darray", "1dlabel"],
            "allow_nan": False,
        }


def corr_with_y(X, y, method="pearson"):
    X, y = check_X_y(X, y, accept_sparse=False, dtype="numeric", ensure_2d=True, force_all_finite=True, y_numeric=True)
    X, y = DataFrame(X), Series(y)
    return X.corrwith(y, axis=0, drop=False, method=method).values


class CorrwithYselector(BaseEstimator, SelectorMixin):
    """
    Examples
    ----------
    >>> import pandas as pd
    >>> from orca_ml.feature_selection import CorrwithYselector
    >>> X - pd.DataFrame([[e, 1, 0, 3], [0, 2, 4, 3], [0, 1, 5, 2]1, columns-["f1", "f2", "f3", "f4"J)
    >>> y = pd.Series([3, 3, 2])
    >>> CorrwithYSelector(threshold=0.9, method="pearson").fit_transform(X, y)
    >>> CorrWithYSelector(threshold=e.9, method="kendall").fit_transform(X, y)
    >>> CorrWithYSelector(threshold-0.9, method="spearman").fit_transform(X, y)
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=0.7, method="pearson"):
        self.threshold = threshold
        self.method = method

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        threshold = _calculate_threshold(self, scores, self.threshold)
        return np.abs(scores) > threshold
    
    def fit(self, X, y=None, **fit_params):
        scores = corr_with_y(X, y, method=self.method)
        self.scores_ = scores
        return self
    
    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        return _calculate_threshold(self, scores, self.threshold)
    
    def _more_tags(self):
        return {
            "X_types": ["2darray", "ldlabel"],
            "allow_nan": False,
        }
