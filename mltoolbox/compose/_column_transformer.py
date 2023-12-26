from inspect import isclass
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.compose._column_transformer import ColumnTransformer as SK_ColumnTransformer
from sklearn.compose._column_transformer import make_column_selector as sk_make_column_selector
from sklearn.utils import _determine_key_type, _get_column_indices
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseEstimator
from ..feature_selection._base import FrameSelectorMixin


class ColumnTransformer(SK_ColumnTransformer):
    def _hstack(self, Xs):
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing everything else from ColumnTransformer.

        Parameters
        -----------
        Xs: list of {array-like, sparse matrix, dataframe}
        """
        if self.sparse_output_:
            try:
                # since all columns should be numeric before stacking them
                # in a sparse matrix, check_array' is used for the
                # dtype conversion if necessary.
                converted_Xs = [check_array(X, accept_sparse=True, force_all_finite=False) for X in Xs]
            except ValueError:
                raise ValueError("For a sparse output, all columns should be a numeric or convertible to a numeric.")
            return sparse.hstack(converted_Xs).tocsr()
        else:
            Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
            if all(hasattr(X, 'iloc') for X in Xs):
                return pd.concat(Xs, axis=1)
            return np.hstack(Xs)

    def _validate_remainder(self, X):
        """Validates remainder and defines _remainder targeting the remaining columns."""
        is_transformer = ((hasattr(self.remainder, "fit") or hasattr(self.remainder, "fit_transform")) and hasattr(self.remainder, "transform"))
        if (self.remainder not in ('drop', 'passthrough') and not is_transformer):
            raise ValueError("The remainder keyword needs to be one of 'drop', 'passthrough', or estimator. '%s' was passed instead" % self.remainder)

        # Make it possible to check for reordered named columns on transform
        self._has_str_cols = any (_determine_key_type(cols) == 'str' for cols in self._columns)
        if hasattr(X, 'columns'):
            self._df_columns = X.columns
        
        self._n_features = X.shape[1]
        cols = []
        column_indices = []
        for columns in self._columns:
            indices = _get_column_indices(X, columns)
            cols.extend(indices)
            column_indices.append(indices)
        
        # add a new attribute `_column_indices` to `ColumnTransformer`
        self._column_indices = column_indices
        remaining_idx = sorted(set(range(self._n_features)) - set(cols))
        self.remainder = ('remainder', self.remainder, remaining_idx or None)


class make_column_selector(sk_make_column_selector, BaseEstimator, FrameSelectorMixin):
    def fit(self, X, y=None, **fit_params):
        if not hasattr(X, 'iloc'):
            raise ValueError("make_column_selector can only be applied to pandas dataframes")
        df_row = X.iloc[:1]
        if self.dtype_include is not None or self.dtype_exclude is not None:
            df_row = df_row.select_dtypes(include=self.dtype_include, exclude=self.dtype_exclude)
        cols = df_row.columns
        if self.pattern is not None:
            cols = cols[cols.str.contains(self.pattern, regex=True)]
        self.cols = cols.tolist()
        self.support_mask_ = X.columns.isin(cols)
        return self
    
    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.cols
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_
    
    def _more_tags(self):
        return {"allow_nan": True}


# change default `_repr_` method for make_column_selector
def _make_column_selector_repr(self):
    kwargs = {
        "pattern": repr(self.pattern),
        "dtype_include": quote(self.dtype_include),
        "dtype_exclude": quote(self.dtype_exclude),
    }
    return "make_column_selector({pattern}, dtype_include={dtype_include}, dtype_exclude={dtype_exclude})".format(**kwargs)


def quote(obj):
    if isinstance(obj, list):
        return "[{}]".format(", ".join(quote(e) for e in obj))
    elif isinstance(obj, set):
        return "set([{}])".format(", ".join(quote(e) for e in obj))
    elif isinstance(obj, tuple):
        return "({})".format(", ".join(quote(e) for e in obj))
    elif isclass(obj):
        if obj.__module__ == 'builtins':
            return obj.__name__
        else:
            return "{}.{}".format(obj.__module__, obj.__name__)
    else:
        return repr(obj)


make_column_selector.__repr__ = _make_column_selector_repr


def make_column_transformer(*transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None, verbose=False):
    """Construct a ColumnTransformer from the given transformers.

    This is a shorthand for the ColumnTransformer constructor; it does not require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow weighting with transformer_weights.

    Read more in the :ref:User Guide <make_column_transformer>.

    Parameters
    -----------
    *transformers : tuples
        Tuples of the form (transformer, columns) specifying the transformer objects to be applied to subsets of the data.

    transformer : {('drop', 'passthrough'} or estimator
        Estimator must support :term:fit and :term: transform Special-cased strings 'drop' and 'passthrough' are accepted as
        well, to indicate to drop the columns or to pass them through untransformed, respectively
    
    columns : str, array-like of str, int, array-like of int, slice, array-like of bool or callable
        Indexes the data on its second axis. Integers are interpreted as positional columns, while strings can reference DataFrame columns
        by name. A scalar string or int should be used where transformer expects x to be a ld array-like (vector), otherwise a 2d array will be passed to the transformer
        A callable is passed the input data x and can return any of the above. To select multiple columns by name or dtype, you can use
        :obj:make_column_selector.
    
    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in transformers are transformed and combined in the output, and the non-specified columns are dropped. (default of 'drop').
        By specifying remainder='passthrough', all remaining columns that were not specified in transformers will be automatically passed
            through. This subset of columns is concatenated with the output of the transformers.
        By setting remainder to be an estimator, the remaining non-specified columns will use the remainder estimator. The estimator must support :term: fit and :term:transform .
    
    sparse_threshold : float, default=0.3
        If the transformed output consists of a mix of sparse and dense data, it will be stacked as a sparse matrix if the density is lower than this
        value. Use sparse_threshold-0 to always return dense. When the transformed output consists of all sparse or all dense data,
        the stacked result will be sparse or dense, respectively, and this keyword will be ignored.
    
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        `None` means 1 unless in a :obj: joblib.parallel_backend' context.
        `-1` means using all processors. See :term: Glossary <n_jobs> for more details.
    
    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be printed as it is completed.
    
    Returns
    -----------
    ct : ColumnTransformer

    See Also
    -----------
    ColumnTransformer : Class that allows combining the outputs of multiple transformer objects used on column subsets of the data into a single feature space
    
    Examples
    -----------
    >>> from mltoolbox.preprocessing import StandardScaler, OneHotEncoder
    >>> from mltoolbox.compose import make_column_transformer
    >>> make_column_transformer((StandardScaler(), ['numerical_column']), (OneHotEncoder(), ['categorical_column']))
    ColumnTransformer(transformers=[('standardscaler', StandardScaler(...), ['numerical_column']), ('onehotencoder', OneHotEncoder(...), ['categorical_column'])])
    """
    # transformer_weights keyword is not passed through because the user would need to know the automatically generated names of the transformers
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(transformer_list, n_jobs=n_jobs, remainder=remainder, sparse_threshold=sparse_threshold, verbose=verbose)


# shortcut methods
def make_passthrough_column_transformer(*transformers, **kwargs):
    kwargs['remainder'] = 'passthrough'
    return make_column_transformer(*transformers, **kwargs)


def make_drop_column_transformer(*transformers, **kwargs):
    kwargs['remainder'] = 'drop'
    return make_column_transformer(*transformers, **kwargs)


def is_column_transformer(obj):
    return issubclass(obj, SK_ColumnTransformer) or isinstance(obj, SK_ColumnTransformer)


def isa_column_transformer(est):
    return isinstance(est, SK_ColumnTransformer)
