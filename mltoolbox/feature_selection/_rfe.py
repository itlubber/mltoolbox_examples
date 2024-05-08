from warnings import warn
import numpy as np
from pandas import DataFrame
from sklearn.feature_selection._rfe import RFE as SK_RFE
from sklearn.feature_selection._rfe import RFECV
from sklearn.utils import safe_mask, check_array
from sklearn.utils._tags import _safe_tags


class RFE(SK_RFE):
    def _get_mask(self, X):
        """Check support and return mask

        Parameters
        ----------
        self : mltoolbox.feature_selection._base.SelectorMixin
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
            if X is pandas DataFrame, return selected column names
            else return selected indices
        """
        data = check_array(X, dtype=None, accept_sparse='csr', force_all_finite=not _safe_tags(self, "allow_nan"))
        mask = self._get_mask(data)
        if isinstance(X, DataFrame):
            return X.columns.values[mask]
        return np.arange(data.shape[1])[mask]
    
    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        self : instance of SelectorMixin
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
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
            x` with columns of zeros inserted where features would have been removed by :meth:`transform`.
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
