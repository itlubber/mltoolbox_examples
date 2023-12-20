import numpy as np
from pandas import DataFrame
from ...base import BaseEstimator, ClassifierMixin, TransformerMixin


class RuleClassifierMixin(ClassifierMixin):
    def score(self, X, y, sample_weight=None):
        """Return the lift score on the given test data and labels.

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -----------
        score : float
            Lift score of self.predict(X) wrt. y.
        """
        from ...metrics._lift_score import lift_score
        return lift_score(y, self.predict(X), sample_weight=sample_weight)


class RuleClassifier(BaseEstimator, RuleClassifierMixin):
    def __init__(self):
        pass

    def fit(self, R, y=None, **fit_params):
        return self
    
    def transform(self, R):
        return DataFrame.from_dict({r.expr: r.result() for r in R})
    
    def predict(self, R):
        X = np.column_stack(tuple(r.result() for r in R))
        return np.any(X, axis=1)
    
    def fit_predict(self, R, y=None, **fit_params):
        return self.fit(R, y=y, **fit_params).predict(R)


class RuleTransformerMixin(TransformerMixin):
    """Mixin class for all transformers which work only on list of Rule.
    
    Notes
    --------
    Subclasses can directly use _check_rules method to confirm a Rule input.
    """
    def fit_transform(self, R, y=None, **fit_params):
        """Fit to data, then transform it.
        
        Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.

        Parameters
        -----------
        R: list of rules, shape (n_rules,)
        y : ndarray of shape (n_samples,), default=None
            Target values.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -----------
        R_new : list of rules, shape (n_rules_new,)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
        # fit method of arity 1 (unsupervised transformation)
            return self.fit(R, **fit_params).transform(R)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(R, y, **fit_params).transform(R)
