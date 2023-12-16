from abc import abstractmethod, ABCMeta
from warnings import warn
import numpy as np
from pandas import DataFrame
from sklearn.feature_selection._base import SelectorMixin as SK_SelectorMixin
from sklearn.utils import check_array
from sklearn.utils import safe_mask
from sklearn.utils._tags import _safe_tags

from ..base import FrameTransformerMixin


class SelectorMixin(SK_SelectorMixin):
    def _get_mask(self, X):
        """Check support and return mask

        Parameters
        ----------
        self : orca_ml.feature_selection._base.SelectorMixin
        X : array of shape (n_samples, n_features)
            The input samples.
        
        Returns
        ----------
        mask : array of shape (n_features,)
            Checked support mask
        """
        mask = self._get_support_mask()
        if not mask.any():
            warn("No features were selected: either the data is too noisy or the selection test too strict.", UserWarning)
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return safe_mask(X, mask)
    
    def get_selected_features(self, X):
        """Get the reduced features on X.

        Parameters
        ----------
        self : sklearn.feature_selection.SelectorMixin
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        features : array of reduced features
            * if X is pandas DataFrame, return selected column names
            * else return selected indices
        """
        data = check_array(X, dtype=None, accept_sparse='csr', force_all_finite=not _safe_tags(self, "allow_nan"))
        mask = self._get_mask(data)
        if isinstance(X, DataFrame):
            return X.columns.values[mask]
        return np.arange(data.shape[1])[mask]
    
    def transform(self, X):
        """Reduce x to the selected features.

        Parameters
        ----------
        self : instance of SelectorMixin
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features
        """
        data = check_array(X, dtype=None, accept_sparse='csr', force_all_finite=not _safe_tags(self, "allow_nan"))
        mask = self._get_mask(data)
        if isinstance(X, DataFrame):
            return X.iloc[:, mask]
        return data[:, mask]
    
    def inverse_transform(self, X):
        """Reverse the transformation operation

        Parameters
        ----------
        self : instance of SelectorMixin
        X : array of shape [n_samples, n_selected_features]
            The input samples.

        Returns
        ----------
        X_r : array of shape [n_samples, n_original_features]
            x with columns of zeros inserted where features would have been removed by :meth: transform .
        """
        if not isinstance(X, DataFrame):
            return super().inverse_transform(X)
        selected_features = X.columns.values
        index = X.index
        support = self._get_support_mask()
        X = check_array(X, dtype=None)
        n_features_in = len(support)
        n_features_out = support.sum()
        if n_features_out != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        columns = np.asarray(["origin%d" % (i + 1) for i in range(n_features_in)], dtype=object)
        columns[support] = selected_features
        Xt = DataFrame(0, index=index, columns=columns)
        Xt[selected_features] = X
        return Xt


class FrameSelectorMixin(FrameTransformerMixin, metaclass=ABCMeta):
    """Transformer mixin that performs feature selection given a support mask

    This mixin provides a feature selector implementation with 'transform' and inverse_transform functionality given an implementation of `_get_support_mask`.
    """

    def get_support(self, indices=False):
        """Get a mask, or integer index, of the features selected

        Parameters
        ----------
        indices : boolean (default False)
                If True, the return value will be an array of integers, rather than a boolean mask.
        
        Returns
        ----------
        support : array
            An index that selects the retained features from a feature vector. If 'indices' is False, this is a boolean array of shape
            [# input features], in which an element is True iff its corresponding feature is selected for retention. If 'indices' is
            True, this is an integer array of shape [# output features] whose values are indices into the input feature vector.
        """
        mask = self._get_support_mask()
        return mask if not indices else np.where(mask)[0]
    
    @abstractmethod
    def _get_support_mask(self):
        """Get the boolean mask indicating which features are selected

        Returns
        ----------
        support : boolean array of shape [# input features]
                An element is True iff its corresponding feature is selected for retention.
        """
    
    def _get_mask(self, X):
        """Check support and return mask

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        mask : array of shape (n_features,)
            Checked support mask
        """
        mask = self._get_support_mask()
        if not mask.any():
            warn( "No features were selected: either the data is too noisy or the selection test too strict.", UserWarning)
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return safe_mask(X, mask)
    
    def get_selected_features(self, X):
        """Get the reduced features on X.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        features : array of reduced features
                The selected column names.
        """
        X = self._ensure_dataframe(X)
        data = check_array(X, dtype=None, accept_sparse='csr', force_all_finite=not _safe_tags(self, key="allow_nan"))
        mask = self._get_mask(data)
        return X.columns.values[mask]
    
    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        ----------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features
        """
        X = self._ensure_dataframe(X)
        data = check_array(X, dtype=None, accept_sparse='csr', force_all_finite=not _safe_tags(self, key="allow_nan"))
        mask = self._get_mask(data)
        return X.iloc[:, mask]
    
    def inverse_transform(self, X):
        """Reverse the transformation operation

        Parameters
        ----------
        X : array of shape [n_samples, n_selected_features]
            The input samples.

        Returns
        ----------
        X_r : array of shape [n_samples, n_original_features]
            X with columns of zeros inserted where features would have been removed by :meth: `transform`.
        """
        X = self._ensure_dataframe(X)
        selected_features = X.columns.values
        index = X.index
        support = self._get_support_mask()

        X = check_array(X, dtype=None)
        n_features_in = len(support)
        n_features_out = support.sum()
        if n_features_out != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        columns = np.asarray(["origin%d" % (i + 1) for i in range(n_features_in)], dtype=object)
        columns[support] = selected_features
        Xt = DataFrame(0, index=index, columns=columns)
        Xt[selected_features] = X
        return Xt

def isa_selector(estimator):
    """Return True if the given estimator is (probably) a selector

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Estimator to test.

    Returns
    ----------
    out : bool
        True if estimator is a selector and False otherwise.
    """
    return isinstance(estimator, (SK_SelectorMixin, FrameSelectorMixin))

def is_selector(obj):
    """Return True if the given object is (probably) a selector.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    ----------
    out : bool
        True if object is a selector and False otherwise.
    """
    return issubclass(obj, (SK_SelectorMixin, FrameSelectorMixin)) or isinstance(obj, (SK_SelectorMixin, FrameSelectorMixin))
