import numpy as np
from pandas import DataFrame
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args
from _base import SelectorMixin
from ..base import BaseEstimator


class ColumnExcluder(BaseEstimator, SelectorMixin):
    """Selector for excluding specific columns from a data

    Parameters
    ----------
    exclude : int str or ld array-like (default: None)
        A list or scalar specifying the feature indices to be excluded. Indices can be string for dataframe input.
        * [1, 4, 5] to exclude the 2nd, sth, and 6th feature columns
        * ['A',"C','D'] to exclude the name of feature columns A, C and D.
        * If None, returns all columns in the array.

    Examples
    ----------
    >>> import pandas as pd
    >>> from orca_m1.feature_selection import ColumnExcluder
    >>> X = pd.DataFrame({"f1": [1, 2], "f2": [3,4], "f3": [4, 5]})
    >>> ColumnExcluder().fit_transform(X)
    >>> ColumnExcluder(exclude="f1").fit_transform(X)
    >>> ColumnExcluder(exclude=[0]).fit_transform(X)
    >>> ColumnExcluder(exclude=0).fit_transform(X)
    >>> ColumnExcluder(exclude=0).fit_transform(X.values)
    array([[3, 4],
        [4,5]], dtype=int64)
    """
    @_deprecate_positional_args
    def __init__(self, *, exclude=None):
        self.exclude = exclude

    def _get_support_mask(self):
        check_is_fitted(self, ["features_", "exclude_"])
        if self.exclude_ is None:
            return np.full(self.features_.shape, True, dtype=bool)
        return np.in1d(self.features_, self.exclude_, invert=True)
    
    def _get_features(self, X, return_int=True):
        check_array(X, dtype=None, force_all_finite=False, ensure_2d=True)
        if return_int:
            return np.arange(X.shape[1])
        else:
            if isinstance(X, DataFrame):
                return X.columns.values
            else:
                raise ValueError("For numpy input, only support using index as feature notation!")
    
    def fit(self, X, y=None, **fit_params):
        self.exclude_, is_int = _calculate_exclude(self.exclude)
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


def _calculate_exclude(exclude):
    if exclude is None:
        return None, False
    if isinstance(exclude, (int, str)):
        exclude = [exclude]
    else:
        exclude = list(exclude)
    types = {type(col) for col in exclude}
    if len(types) > 1:
        raise ValueError("All elements in exclude should be all of the same type.")
    typ = list(types)[0]
    if typ is not int and typ is not str:
        raise ValueError("All elements in exclude should be int or str. Got {}".format(typ))
    return exclude, typ is int
