import numpy as np
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted, check_array
from ._base import SelectorMixin
from ..base import BaseEstimator


class FeatureSetSelector (BaseEstimator, SelectorMixin):
    """Feature selection via a set of features.
    
    Examples
    -----------
    >>> import pandas as pd
    >>> from mltoolbox.feature_selection import FeatureSetSelector
    >>> X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "f3": [4, 5]})
    >>> FeatureSetSelector().fit_transform(X)
    >>> FeatureSetSelector (subset="f1").fit_transform(X)
    >>> FeatureSetSelector(subset=[0]).fit_transform(X)
    >>> FeatureSetSelector(subset=0).fit_transform(X)
    >>> FeatureSetSelector(subset-0).fit transform(X.values)
    array([[1],
           [2]], dtype=int64)
    """
    def __init__(self, subset=None):
        self.subset = subset

    def _get_support_mask(self):
        check_is_fitted(self, ["features_", "subset_"])
        if self.subset_ is None:
            return np.full(self.features_.shape, True, dtype=bool)
        return np.in1d(self.features_, self.subset_, invert=False)
    
    def _get_features(self, X, return_int=True):
        check_array(X, dtype=None, force_all_finite=False, ensure_2d=True)
        if return_int:
            return np.arange(X.shape[1])
        else:
            if isinstance(X, DataFrame):
                return X.columns.values
            else:
                raise ValueError("For numpy input, only support using index as feature notation!")
        
    def fit(self, X, y=None):
        self.subset_, is_int = _calculate_cols(self.subset)
        self.features_ = self._get_features(X, is_int)
        return self
    
    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self._get_support_mask()
    
    def _more_tags(self):
        return {
            "X_types": ["2darray", "categorical", "string"],
            "allow_nan": True,
        }


def _calculate_cols(cols):
    if cols is None:
        return None, False
    if isinstance(cols, (int, str)):
        cols = [cols]
    else:
        cols = list(cols)
    types = {type(col) for col in cols}
    if len(types) > 1:
        raise ValueError("All elements in exclude should be all of the same type.")
    typ = list(types)[0]
    if typ is not int and typ is not str:
        raise ValueError("All elements in exclude should be int or str. Got {}".format(typ))
    return cols, typ is int
