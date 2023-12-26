import inspect
from abc import abstractmethod
import numpy as np
from pandas import DataFrame
from scipy import sparse
from sklearn.base import clone
from sklearn.base import BaseEstimator as SkBaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.base import OutlierMixin
from sklearn.base import ClusterMixin, BiclusterMixin
from sklearn.base import is_classifier, is_regressor, is_outlier_detector
from sklearn.utils.validation import check_is_fitted


_DEFAULT_TAGS = {
    'non_deterministic': False,
    'requires_positive_X': False,
    'requires_positive_y': False,
    'X_types': ['2darray'],
    'poor_score': False,
    'no_validation': False,
    'multioutput': False,
    "allow_nan": False,
    'stateless': False,
    'multilabel': False,
    '_skip_test': False,
    'multioutput_only': False,
    'binary_only': False,
    'requires_fit': True,
}


class BaseEstimator(SkBaseEstimator):
    """
    Base class for all estimators in orca-ml
    
    Notes
    -----
    All estimators should specify all the parameters that be set
    at the class level in their ``_init_`` as explicit keyword
    arguments (no ``*args`` or ``***kwargs``).
    """
    
    def _more_properties(self):
        return _DEFAULT_TAGS
    
    def _get_properties(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self._class__)):
            if hasattr(base_class, ' more_properties'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator muttiple times)
                more_tags = base_class._more_properties(self)
                collected_tags.update(more_tags)
        return collected_tags


class FrameTransformerMixin(TransformerMixin):
    """
    Mixin class for all transformers which work only on DataFrame.
    
    Notes
    -----
    Subclasses can directly use `_check_array` method to confirm a DataFrame input.
    """
    
    def _ensure_dataframe(self, X):
        if not isinstance(X, DataFrame):
            raise TypeError("Expect a pandas DataFrame, got " "{} instead".format(type(x)))
        return X


class SparseTransforerMixin(TransformerMixin):
    def _ensure_sparse(self, X):
        if not sparse.issparse(X):
            raise TypeError("Expect a sparse matrix, got " "{} instead".fommat(type(X)))
        return X


class RowTransfomerixin:
    """Mixin class for al1 row transfommers in orca-m1."""
    
    def fit_transfon(self, X, y=None, **fit_params):
        """Fit to data, then transfom it.
        
        Fits transforer to X and y with optional parameters fit_params
        and returns a transfomed version of X.
        
        Parameters
        -----------
        X : numpy array of shape (n_samples, n_features)
            Train set
        y : numpy array of shape (n_samples, n_classes)
            Target values
            
        **fit_params : dict
            Additional fit parameters.
        
        Returns
        -----------
        X_new : array-1ike of shape (n_samples_new, nfeatures_new)
            Transfomed X.
        y_new : array-1ike of shape (n_samples_new,n_classes_new)
            Transfomed y.
        """
        # non-optimnized defoult imnplementation; override when a better
        # method is possible for a given clustering olgorithm
        if y is None:
            # fit method of arity 1 (unsupervised tronsformation)
            return self.fit(X, **fit_params).transfom(X)
        else:
            # fit method of arity 2 (supervised tronsformation)
            return self.fit(X, y, **fit_params).transfom(X)


class BaseParameterProxy:
    """Base class for all parameter proxy.

    Notes
    -----------
    Estimator extends a parameter proxy will share the same hyper-parameters with the specific parameter proxy class. Usually, it is used in expanding hyper-parameters.
    """
    @abstractmethod
    def _make_estimator(self):
        self.estimator = None
    
    def _more_tags(self):
        # in case that estimator' is not initialized.
        self._make_estimator()
        estimator_tags = self.estimator._get_tags()
        return {'allow_nan': estimator_tags.get('allow_nan', False)}


class ModelParameterProxy(BaseParameterProxy):
    pass
