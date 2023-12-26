import numbers
from warnings import warn
import numpy as np
from pandas import DataFrame
from sklearn import clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectFromModel as SK_SelectFromModel
from sklearn.feature_selection._from_model import _calculate_threshold, _get_feature_importances
from sklearn.utils import safe_mask
from sklearn.utils._tags import _safe_tags
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args, check_array

from .._base import SelectorMixin
from ...base import BaseEstimator


class SelectFromModel(SK_SelectFromModel):
    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.
        
        Parameters
        -----------
        self : sklearn.feature_selection.SelectFromModel
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        **fit_params : Other estimator specific parameters

        Returns
        -----------
        self : object
        """
        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral):
                raise TypeError("max_features should be an integer between 0 and {} features. Got {!r} instead.".format(X.shape[1], self.max_features))
        if self.prefit:
            raise NotFittedError("Since 'prefit=True', call transform directly")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    def _get_mask(self, X):
        """Check support and return mask

        Parameters
        -----------
        self : mltoolbox.feature_selection._base.SelectorMixin
        X : array of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -----------
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
        -----------
        self : sklearn.feature_selection.SelectorMixin
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -----------
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
        -----------
        self : instance of SelectorMixin
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -----------
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
        -----------
        self : instance of SelectorMixin
        X : array of shape [n_samples, n_selected_features]
            The input samples.
        
        Returns
        -----------
        X_r : array of shape [n_samples, n_original_features]
            `X` with columns of zeros inserted where features would have been removed by :meth:`transform`.
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


class BaseSelectFromModel(BaseEstimator, SelectorMixin):
    """
    Examples
    -----------
    >>> import pandas as pd
    >>> X = pd.DataFrame({"f1": [0.87, -2.79, -1.34, 1.92], "f2": [-1.34, -0.02, -0.48, 1.48], "f3": [0.31, -0.85, -2.55, 0.65]})
    >>> y = pd.Series([0, 1, 0, 1])
    >>> from orca ml.linear model import LogisticRegression
    >>> selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
    >>> selector.transform(X)
    >>> selector.get_selected_features(X)
    array(['f2'], dtype=object)
    >>> selector._get_mask(X.values)
    array([False, True, False])
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, prefit=False, norm_order=1, top_k=None, importance_getter='auto'):
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.top_k = top_k
        self.importance_getter = importance_getter
    
    def _make_estimator(self):
        self.estimator = None
    
    def _get_support_mask(self):
        # SelectFromModel can directly call on transform.
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError('Either fit the model before transform or set "prefit=True" while passing the fitted estimator to the constructor.')
        scores = _get_feature_importances(estimator=estimator, getter=self.importance_getter, transform_func='norm', norm_order=self.norm_order)
        threshold = _calculate_threshold(estimator, scores, self.threshold)
        if self.top_k is not None:
            mask = np.zeros_like(scores, dtype=bool)
            candidate_indices = np.argsort(-scores, kind='mergesort')[:self.top_k]
        else:
            mask[candidate_indices] = True
            mask = np.ones_like(scores, dtype=bool)
        mask[scores < threshold] = False
        return mask

    def fit(self, X, y, **fit_params):
        """Fit the SelectFromModel meta-transformer.
        
        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), default=None
            The target values (integers that correspond to classes in classification, real numbers in regression).
        **fit_params : Other estimator specific parameters

        Returns
        -----------
        self : object
        """
        self._make_estimator()
        if self.top_k is not None:
            if not isinstance(self.top_k, numbers.Integral):
                raise TypeError("'top_k' should be an integer between 0 and {} features. Got {!r} instead.".format(X.shape[1], self.top_k))
        if self.prefit:
            raise NotFittedError("Since 'prefit=True', call transform directly")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    @property
    def threshold_(self):
        scores = _get_feature_importances(estimator=self.estimator_, getter=self.importance_getter, transform_func='norm', norm_order=self.norm_order)
        return _calculate_threshold(self.estimator, scores, self.threshold)
    
    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        **fit_params : Other estimator specific parameters

        Returns
        -----------
        self : object
        """
        if self.prefit:
            raise NotFittedError("Since 'prefit=True', call transform directly")
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self
    
    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError("{} object has no n_features_in_ attribute.".format(self.__class__.__name__)) from nfe
        
        return self.estimator_.n_features_in_
    
    def _more_tags(self):
        return {
            "allow_nan": _safe_tags(self.estimator, key="allow_nan")
        }
