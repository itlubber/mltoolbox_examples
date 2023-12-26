import numpy as np
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args
from ._base import SelectorMixin
from ..base import BaseEstimator


class ValueRatioFilter(BaseEstimator, SelectorMixin):
    """
    Examples
    ---------
    >>> import pandas as pd
    >>> from orca ml feature selection import ValueRatiofilter
    >>> X = pd.DataFrame([[0, -99, 0, 3], [0, -99, 4, 3], [0, 1, -99, 3]], columns=["f1", "f2", "f3", "f4"])
    >>> selector = ValueRatioFilter(values=-99, threshold="mean").fit(X)
    >>> selector.transform(X)
    """
    @_deprecate_positional_args
    def __init__(self, *, values=-99, threshold="mean"):
        self.values = values
        self.threshold = threshold

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        threshold = _calculate_threshold(self, scores, self.threshold)
        return scores <= threshold
    
    def fit(self, X, y=None, **fit_params):
        values = self.values
        if isinstance(values, (int, float, str)):
            values = [values]
        else:
            values = list(values)
        scores = _value_ratios(X, values)
        self.scores_ = scores
        return self
    
    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)
    
    # accept data with missing values
    def _more_tags(self):
        return {
            "X_types": ["2darray", "categorical", "string"],
            "allow_nan": False,
        }


def _value_ratios(X, values=None):
    X = check_array(X, dtype=None, ensure_2d=True, force_all_finite=True)
    return np.mean(np.isin(X, values), axis=0)


def value_ratios(X, values=None):
    selector = ValueRatioFilter(values=values)
    selector.fit(X)
    return selector.scores_
