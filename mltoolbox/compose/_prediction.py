import warnings
import numpy as np
from pandas import DataFrame
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import _safe_indexing
# from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator, ClassifierMixin, clone, if_delegate_has_method
from ..postprocessing import BaseScoreTransformer


class TransformedPredictionClassifier(BaseEstimator, ClassifierMixin):
    """Meta-estimator to transform on a classifier's predictions.

    Useful for applying a non-linear transformation to the prediction 'y_pred in
    classification results. This transformation can be given as a Transformer
    such as the StandScoreTransformer
    """
    def __init__(self, classifier=None, *, transformer=None, func=None, inverse_func=None, check_inverse=True):
        self.classifier = classifier
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
    
    def _fit_transformer(self, y):
        """
        Check transformer and fit transformer
        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).
        """
        if (self.transformer is not None and (self.func is not None or self.inverse_func is not None)):
            raise ValueError("'transformer' and functions 'func'/'inverse func' cannot both be set.")
        elif self.transformer is not None:
            self.transformer = clone(self.transformer)
        else:
            if self.func is not None and self.inverse_func is None:
                raise ValueError("When 'func' is provided, 'inverse_func' must also be provided")
            self.transformer_ = FunctionTransformer(func=self.func, inverse_func=self.inverse_func, validate=True, check_inverse=self.check_inverse)

        # XXX: sample_weight is not currently passed to the
        # transformer. However, if transformer starts using sample_weight, the
        # code shoutd be modified accordingly. At the time to consider the
        # sample_prop feature, itis also a good use case to be considered.
        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = _safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)
            if not np.allclose(y_sel, self.transformer_.inverse_transform(y_sel_t)):
                warnings.warn("The provided functions or transformer are not strictly inverse of each other. If you are sure you want to proceed regardless, set 'check_inverse=False'", UserWarning)
    
    def fit(self, X, y, **fit_params):
        """
        Fit underlying classifier on train data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Train data
        y : array-like, shape (n_samples,)
            Train label
        fit_params : dict
            Fit parameters

        Returns
        ----------
        self
        """
        
        if self.classifier is None:
            from ..linear_model import LogisticRegression
            self.classifier_ = LogisticRegression()
        else:
            self.classifier_ = clone(self.classifier)

        if self.classifier_._estimator_type != 'classifier':
            raise ValueError("Should be a classifier!")
        
        self.classifier_.fit(X, y, **fit_params)

        y_pred = self.classifier_.predict_proba(X)[:, 1]
        self._fit_transformer(y_pred[:, None])
        return self

    def _transform(self, X):
        """Transform predict probability to scores
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data

        Returns
        ----------
        score : array-like, shape (n_samples,)
            The transformed score
        """
        check_is_fitted(self)
        y_pred = self.classifier_.predict_proba(X)[:, 1]
        return self.transformer_.transform(y_pred[:, None])
        
    def transform(self, X):
        data = self._transform(X)
        if isinstance(X, DataFrame):
            return DataFrame(data=data, columns=[self._class_._name__], index=X.index)
        return data

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to x and y with optional parameters  fit_params) and returns a transformed version of x.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        у : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.
        
        Returns
        ----------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized defautt implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)
        
    @if_delegate_has_method(delegate="classifier_")
    def predict(self, X):
        """Estimate the best class label for each sample in X.

        Parameters
        ---------
        X : array-like of shape (n_samples, n_features)
            The data

        Returns
        ---------
        y : array-like of shape (n_samples,)
            Predicted label for each sample in X.
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict(X)
    
    def fit_predict(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.predict(X)
    
    @if_delegate_has_method(delegate="classifier_")
    def predict_proba(self, X):
        """Get predict probabilities by inner classifier.

        Parameters
        ---------
        X : array-like, shape (n_samples, n_features)
            The data

        Returns
        ---------
        prob : array-like, shape (n_samples, 2)
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_proba(X)
    

    @if_delegate_has_method(delegate="classifier_")
    def _predict_proba_lr(self, X):
        """Get predict probabilities by inner classifier.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data

        Returns
        ----------
        prob : array-like, shape (n_samples, 2)
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_._predict_proba_lr(X)
    
    @if_delegate_has_method(delegate="classifier_")
    def predict_log_proba(self, X):
        """Get predict log probabilities by inner classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data

        Returns
        ----------
        log_prob : array-like, shape (n_samples, 2)
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.predict_log_proba(X)
    
    @if_delegate_has_method(delegate="classifier_")
    def decision_function(self, X):
        """Decision function for the scorecard.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data

        Returns
        ----------
        y : array-like of shape (n_samples, 1)
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.decision_function(X)
    
    @if_delegate_has_method(delegate="classifier_")
    def score_samples(self, X):
        """Return score_samples of the final estimator.

        Parameters
        ----------
        X: array-like

        Returns
        ----------
        Y_score : ndarray, shape (n_samples,)
        """
        check_is_fitted(self, "classifier_")
        return self.classifier_.score_samples(X)

    @property
    def classes_(self):
        check_is_fitted(self, "classifier_")
        return self.classifier_.classes_
    
    @property
    def n_classes_(self):
        check_is_fitted(self, "classifier_")
        return len(self.classifier_.classes_)
    
    @if_delegate_has_method(delegate="estimator_")
    def score_samples_(self):
        check_is_fitted(self, "classifier_")
        return self.classifier_.score_samples_
    
    @property
    def _pariwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix."""
        return getattr(self.classifier_, "_pairwise", False)
    
    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}
    
    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() returns False the estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError("{} object has no n_features_in_ attribute.".format(self._class___name__)) from nfe
        
        return self.classifier_.n_features_in_


class ScoreCard(TransformedPredictionClassifier):
    """
    ScoreCard class providing basic fit, transform, predict, output usage.
    """
    def __init__(self, classifier=None, *, transformer=None, func=None, inverse_func=None, check_inverse=False):
        super(ScoreCard, self).__init__(classifier=classifier, transformer=transformer, func=func, inverse_func=inverse_func, check_inverse=check_inverse)
    
    def predict(self, X):
        check_is_fitted(self, ["classifier_", "transformer_"])
        if not isinstance(self.transformer_, BaseScoreTransformer):
            raise ValueError("For ScoreCard application, `transformer` should be a `BaseScoreTransformer`")
        classes_ = self.classifier_.classes
        y_pred = self.classifier_.predict_proba(X)[:, -1]
        indices = self.transformer_.predict(y_pred[:, None])
        return classes_[indices]

    def score(self, X, y, sample_weight=None):
        """
        Return the roc auc score on the given test data and labels

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for 'x
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights
            
        Returns
        ----------
        score : float
            Roc auc score of self.predict_proba(X)[:, -1] wrt. y.
        """
        from sklearn.metrics._ranking import roc_auc_score
        return roc_auc_score(y, self.predict_proba(X) [ :, -1], sample_weight=sample_weight)


def is_scorecard(obj):
    return issubclass(obj, ScoreCard) or isinstance(obj, ScoreCard)


def isa_scorecard(est):
    return isinstance(est, ScoreCard)
