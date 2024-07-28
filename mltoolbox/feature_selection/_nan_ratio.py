import numpy as np
from scipy import sparse
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.impute._base import _check_inputs_dtype
from sklearn.utils import is_scalar_nan
from sklearn.utils._mask import _get_mask
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args
from ._base import SelectorMixin
from ..base import BaseEstimator


class NanFilter(BaseEstimator, SelectorMixin):
    """Nan ratio filter for missing values

    Note that this component typically should be used in preprocessing stage

    Parameters
    ----------
    threshold : float, int or str, default=0.95
        The threshold value to use for feature selection. Features whose missing ratio is greater are filtered while the others are
        kept. If "median" (resp. "mean"), then the threshold value is the median (resp. the mean) of the feature missing ratios. A scaling
        factor (e.g., *1.25*mean") may also be used. If None and if the estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is le-5. Otherwise, "mean" is used by default.
    
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of missing values will be summarized to a ratio.
    sparse : boolean or "auto", default="auto"
        Whether the imputer mask format should be sparse or dense

        - If "auto" (default), the imputer mask will be of same type as input.
        - If True, the imputer mask vill be a sparse matrix.
        - If False, the imputer mask will be a numpy array.

    Attributes
    ----------
    scores_ : ndarray, shape (n_features,)
        The features nan ratios. They are computed during fit

    Examples
    ----------
    >>> import pandas as pd
    >>> from mltoolbox.feature_selection._nan_ratio import NanFilter
    >>> x = pd.DataFrame([[0, np.nan, 0, 3],[0, None,4, 3], [0, 1, np.nan, 3]], columns=["f1", "f2", "f3", "f4"])
    >>> selector = NanFilter(threshold="median").fit(x)
    NanFilter(missing_values=nan, sparse='auto', threshold='median')
    >>> xt = selector.transform(x)
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=0.95, missing_values=np.nan, sparse="auto"):
        self.threshold = threshold
        self.missing_values = missing_values
        self.sparse = sparse
    
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        threshold = _calculate_threshold(self, scores, self.threshold)
        return scores <= threshold
    
    def _get_imputer_mask(self, X):
        """Compute the imputer mask and the indices of the features containing missing values.

        Parameters
        ----------
        X : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The input data with missing values. Note thatx has been checked in fit and transform before to call this function.

        Returns
        ----------
        imputer_mask : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The imputer mask of the original data.
        features_with_missing : ndarray, shape (n_features_with_missing)
            The features containing missing values.
        """
        if sparse.issparse(X):
            mask = _get_mask(X.data, self.missing_values)
            # The imputer mask will be constructed with the same sparse format as X.
            sparse_constructor = (sparse.csr_matrix if X.format == 'csr' else sparse.csc_matrix)
            imputer_mask = sparse_constructor((mask, X.indices.copy(), X.indptr.copy()), shape=X.shape, dtype=bool)
            imputer_mask.eliminate_zeros()

            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == 'csr':
                imputer_mask = imputer_mask.tocsc()
        else:
            imputer_mask = _get_mask(X, self.missing_values)

            if self.sparse is True:
                imputer_mask = sparse.csc_matrix(imputer_mask)

        return imputer_mask
    
    def _validate_input(self, X):
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        X = check_array(X, accept_sparse=('csc', 'csr'), dtype=None, force_all_finite=force_all_finite)
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("MissingIndicator does not support data with dtype {0}. Please provide either a numeric array"
                            " (with a floating point or integer dtype) or categorical data represented either as an array "
                            "with integer dtype or an array of string values with an object dtype.".format(X.dtype))
        
        if sparse.issparse(X) and self.missing_values == 0:
            # missing_values = 0 not allowed with sparse data as it would force densification
            raise ValueError("Sparse input with missing_values=0 is not supported. Provide a dense array instead.")
        return X
    
    def fit(self, X, y=None):
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and n_features is the number of features
        y : Ignore
            Passsthrough for pipeline compatibility

        Returns
        ----------
        imputer_mask : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The imputer mask of the original data
        """
        X = self._validate_input(X)
        if not ((isinstance(self.sparse, str) and self.sparse == "auto") or isinstance(self.sparse, bool)):
            raise ValueError("sparse has to be a boolean or 'auto'. Got {!r} instead.".format(self.sparse))
        imputer_mask = self._get_imputer_mask(X)
        self.scores_ = np.mean(imputer_mask, axis=0)
        return self

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)
    
    # aсcept data with missing values
    def _more_tags(self):
        return {
            "X_types": ["2darray", "categorical", "string"],
            "allow_nan": True,
        }


def missing_ratios(X, y=None, missing_values=np.nan):
    """
    Parameters
    ----------
    X : array-like
    y : array-like
    missing_values : Real, default=np.nan
    """
    selector = NanFilter(missing_values=missing_values)
    selector.fit(X, y)
    return selector.scores_


nan_ratios = missing_ratios
