import numpy as np
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_string_dtype, is_object_dtype
from sklearn.utils.validation import check_is_fitted
from ._base import FrameSelectorMixin
from ..base import BaseEstimator


def _is_category_dtype(x):
    return is_categorical_dtype(x) or is_object_dtype(x) or is_string_dtype(x)


class CategoricalDtypeSelector(BaseEstimator, FrameSelectorMixin):
    """
    Examples
    ---------
    >>> import pandas as pd
    >>> from orca_ml.feature_selection import CategoricalDtypeSelector
    >>> X = pd.DataFrame([[0, 2, 0, 'b'], [0, 1, np.nan, 'a'], [1, 2, 3, 'c']], columns=["f1", "f2", "f3", "f4"])
    >>> selector = CategoricalDtypeSelector().fit(x)
    >>> selector.transform(X)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        X = self._ensure_dataframe(X)
        _, n_features = X.shape
        mask = np.fromiter((is_categorical_dtype(X.iloc[:, i]) for i in range(n_features)), dtype=bool)
        self.support_mask_ = mask
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_
    
    def _more_tags(self):
        return {
            'X_types': ['categorical'],
            'allow nan': True,
        }


class ObjectDtypeSelector(BaseEstimator, FrameSelectorMixin):
    """
    Examples
    ---------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from orca_m1l.feature_selection.type_selector import ObjectDtypeSelector
    >>> X = pd.DataFrame([[e, 2, 0, 'b'], [e, 1, np.nan, 'a'], [1, 2, 3, 'c"]], columns=["f1", "f2", "f3", "f4"])
    >>> selector = ObjectDtypeSelector().fit(x)
    >>> selector.transform(X)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        X = self._ensure_dataframe(X)
        _, n_features = X.shape
        mask = np.fromiter((is_object_dtype(X.iloc[:, i]) for i in range(n_features)), dtype=bool)
        self.support_mask_ = mask
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_
    
    def _more_tags(self):
        return {
            'X_types': ['2darray', 'object'],
            'allow_nan': True,
        }


class StringDtypeSelector(BaseEstimator, FrameSelectorMixin):
    """
    Examples
    -----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from orca_ml.feature_selection.type_selector import StringDtypeSelector
    >>> X = pd.DataFrame([[0, 2, 0, 'b'], [0, 1, np.nan, 'a'l, [1, 2, 3, 'c']], columns=["f1", "f2", "f3", "f4"])
    >>> selector = StringDtypeSelector().fit(x)
    >>> selector.transform(X)
    """
    def __init__(self):
        pass

    def fit(self, x, y=None, **fit_params):
        X = self._ensure_dataframe(X)
        _, n_features = X.shape
        mask = np.fromiter((is_string_dtype(X.iloc[:, i]) for i in range(n_features)), dtype=bool)
        self.support_mask_ = mask
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_
    
    def _more_tags(self):
        return {
            'X_types': ['2darray', 'string'],
            'allow_nan': True,
        }


class NumericDtypeSelector(BaseEstimator, FrameSelectorMixin):
    """
    Examples
    ----------
    >>> import numpy as np
    >>> import pandas as po
    >>> from orca_ml.feature_selection.type_selector import NumericDtypeSelector
    >>> x = pd.DataFrame([[0, 2, 0, 'b'], [0, 1, пp.nan, 'a'], [1, 2, 3, 'с']], columns=["f1", "f2", "f3", "f4"])
    >> selector = NumericDtypeSelector().fit(X)
    >>>
    selector.transform(X)
    """
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        X = self._ensure_dataframe(X)
        _, n_features = X.shape
        mask = np.fromiter((is_numeric_dtype(X.iloc[:, i]) for i in range(n_features)), dtype=bool)
        self.support_mask_ = mask
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_

    def _get_tags(self):
        return {
            'X_types': ['2darray'],
            'allow_nan': True,
        }

class MakeColumnSelector(BaseEstimator, FrameSelectorMixin):
    def __init__(self, pattern=None, *, dtype_include=None, dtype_exclude=None):
        self.pattern = pattern
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def fit(self, X, y=None, **fit_params):
        if not hasattr(X, 'iloc'):
            raise ValueError("make_column_selector can only be applied to pandas dataframes")
        df_row = X.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(include=self.dtype_include, exclude=self.dtype_exclude)
        cols = df_row.columns
        if self.pattern is not None:
            cols = cols[cols.str.contains(self.pattern, regex=True)]
        self.cols_ = cols.tolist()
        self.support_mask_ = X.columns.isin(cols)
        return self
    
    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.cols_
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_
    
    def _more_tags(self):
        return {
            "allow_nan": True,
        }


# select all numeric dtypes
num = MakeColumnSelector(dtype_include="number")
not_num = MakeColumnSelector(dtype_exclude="number")

# select all object dtypes
obj = MakeColumnSelector(dtype_include="object")
not_obj = MakeColumnSelector(dtype_exclude="object")

# select categorical dtypes
cat = MakeColumnSelector(dtype_include="category")
not_cat = MakeColumnSelector(dtype_exclude="category")

# datetime
datetime = MakeColumnSelector(dtype_include="datetime")
not_datetime = MakeColumnSelector(dtype_exclude="datetime")

# timedelta
timedelta = MakeColumnSelector(dtype_include="timedelta")
not_timedelta = MakeColumnSelector(dtype_exclude="timedelta")

# datetimez
datetimez = MakeColumnSelector(dtype_include="datetimez")
not_datetimez = MakeColumnSelector (dtype_exclude="datetimez")
