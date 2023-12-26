from pandas import DataFrame
from sklearn.utils.validation import check_array, check_is_fitted, _deprecate_positional_args
from ._base import SelectorMixin
from ..base import BaseEstimator


class CardinalityFilter(BaseEstimator, SelectorMixin):
    """Feature selection via categorical feature's cardinality.

    Examples
    -----------
    >>> import pandas as pd
    >>> x = pd.DataFrame({"f2": ["F", "м", "F"], "f3": ["M1", "M2", "м3"]})
    >>> from mltoolbox.feature_selection import CardinalityFilter
    >>> cs = CardinalityFilter(threshold=2)
    >>> cs.fit transform(X)
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=10, dropna=True):
        self.threshold = threshold
        self.dropna = dropna
    
    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        return self.scores_ <= self.threshold
    
    def fit(self, X, y=None, **fit_params):
        X = check_array(X, dtype=object, ensure_2d=True, force_all_finite="allow-nan", ensure_min_features=1)
        self.scores_ = col_uniques(X, self.dropna)
        return self
    
    def _more_tags(self):
        return {
            "X_types": ["categorical", "string"],
            "allow_nan": True,
        }


def col_uniques(X, dropna=True):
    """Count all column's unique value

    Parameters
    -----------
    X : array-like
    dropna : bool
    """
    if not hasattr(X, 'iloc'):
        X = DataFrame(X)
    return X.nunique(axis=0, dropna=dropna).values


def cardinality_scores(X, dropna=True):
    selector = CardinalityFilter(dropna=dropna)
    selector.fit(X)
    return selector.scores_
