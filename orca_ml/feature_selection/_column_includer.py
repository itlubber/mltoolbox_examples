import numpy as np
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args
from ._base import SelectorMixin
from ..base import BaseEstimator


# borrowed from mlxtend
class ColumnIncluder (BaseEstimator, SelectorMixin):
    """Selector for including specific columns from a data.

    Parameters
    ----------
    include : int str or ld array-like (default: None)
        A list or scalar specifying the feature indices to be included. Indices can be string for dataframe input.
        * [1, 4, 5] to include the 2nd, 5th, and 6th feature columns
        * ['A', 'C', 'D'] to include the name of feature columns A, C and D.
        * If None, returns all columns in the array.

    Examples
    ----------
    >>> import pandas as pd
    >>> from orca_ml.feature_selection import ColumnIncluder
    >>> x = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "f3": [4, 5]})
    >>> ColumnIncluder().fit_transform(x)
    >>> ColumnIncluder(include="f1").fit_transform(X)
    >>> ColumnIncluder(include=[0]).fit_transform(X)
    >>> ColumnIncluder(include=0).fit_transform(X)
    >>> ColumnIncluder(include=0).fit_transform(X.values)
    array([[1],
           [2]], dtype=int64)
    """
    @_deprecate_positional_args
    def __init__(self, *, include=None):
        self.include = include

    def _get_support_mask(self):
        check_is_fitted(self, ["features_", "include_"])
        if self.include_ is None:
            return np.full(self.features_.shape, True, dtype=bool)
        return np.in1d(self.features_, self.include_, invert=False)
    
    def _get_features(self, X, return_int=True):
        check_array(X, dtype=None, force_all_finite=False, ensure_2d=True)
        if return_int:
            return np.arange(X. shape[1])
        else:
            if isinstance(X, DataFrame):
                return X.columns.values
            else:
                raise ValueError("For numpy input, only support using index as feature notation!")
    
    def fit(self, X, y=None, **fit_params):
        self.include_, is_int = _calculate_include(self.include)
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


def _calculate_include(include):
    if include is None:
        return None, False
    if isinstance(include, (int, str)):
        include = [include]
    else:
        include = list(include)
    types = {type(col) for col in include}
    if len(types) > 1:
        raise ValueError("All elements in exclude should be all of the same type.")
    typ = list(types)[0]
    if typ is not int and typ is not str:
        raise ValueError("All elements in exclude should be int or str. Got {}".format(typ))
    return include, typ is int
