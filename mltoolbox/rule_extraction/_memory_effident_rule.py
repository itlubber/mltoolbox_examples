from enum import Enum

import numexpr as ne
import numpy as np
from pandas import DataFrame
from sklearn.utils import check_array
from ._rule_predict_result import RulePredictResult


def _get_context(X, feature_names):
    """Make context for code evaluation.

    Parameters
    -----------
    X : array of shape (n_samples, n_features)
        DataFrame should be converted earlier.
    feature_names : array-like of shape (n_features, )
    
    Returns
    ctx : dict, evaluation context
    """
    return {name: X[:, i] for i, name in enumerate(feature_names)}


def _apply_expr_on_array(expr, X, feature_names):
    ctx = _get_context(X, feature_names)
    return ne.evaluate(expr, local_dict=ctx)


class RuleState(str, Enum):
    INITIALIZED = "initialized"
    APPLIED = "applied"


class RuleStateError(RuntimeError):
    pass


class RuleUnAppliedError(RuleStateError):
    pass


class Rule:
    """Executable rule for decision making.
    
    Parameters
    -----------
    expr : str
        The rule expression.
    
    Examples
    -----------
    >>> import pandas as pd
    >>> from mltoolbox.rule_extraction._rule import Rule
    >>> x = pd.DataFrame.from_dict({"f1": [1, 2, 3, 4], "f2": [2, 4, 6, 8]})
    >>> r0 = Rule("f1 > 2")
    >>> r1 = Rule("f2 < 5")
    >>> r0.predict(x)
    array([False, False, True, True])
    >>> r1.predict(x)
    array([True, True, False, False])
    >>> (r0 | r1).predict(x)
    array([True, True, True, True])
    >>> (r0 & r1).predict()
    array([False, False, False, False])
    """
    def __init__(self, expr):
        self.expr = expr
        self._state = RuleState.INITIALIZED
    
    def __str__(self):
        return f"Rule({repr(self.expr)})"
    
    def __repr__(self):
        return f"Rule({repr(self.expr)})"
    
    def predict(self, x):
        """Apply rule on the specific data.

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
            Data used for applying rules.

        Returns
        -----------
        y_pred : array-like of shape (n_samples, )
            Boolean column represents the rule predicting result.
        """
        if self._state == RuleState.INITIALIZED:
            if isinstance(X, DataFrame):
                feature_names = x.columns.values
            else:
                feature_names = np.fromiter(("f{}" % i for i in range(X. shape[1])), dtype=object)
            X = check_array(X, dtype="numeric")
            mask = _apply_expr_on_array(self.expr, X, feature_names)
            self._result = RulePredictResult.from_dense(mask)
            self._state = RuleState.APPLIED
        return self._result
    
    @property
    def result(self):
        if self._state != RuleState.APPLIED:
            raise RuleUnAppliedError("Invoke predict to make a rule applied.")
        return self.result
    
    def __eq__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        same_expr = self.expr == other.expr
        if self._state == RuleState.INITIALIZED:
            return same_expr
        same_result = self.result == other.result
        return same_expr and same_result
    
    # rule combinations
    def __or__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        r = Rule(f"({self.expr}) | ({other.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r._result = self.result | other.result
        r._state = RuleState.APPLIED
        return r
    
    def __and__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        r = Rule(f"({self.expr}) & ({other.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r._result = self.result & other.result
        r._state = RuleState.APPLIED
        return r
    
    def __xor__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        r = Rule(f"({self.expr}) ^ ({other.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r._result = self.result ^ other.result
        np.logical_xor()
        r._state = RuleState.APPLIED
        return r
    
    def __mul__(self, other):
        return self.__or__(other)
    
    def __invert__(self):
        r = Rule(f"~({self.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r._result = ~self.result
        r._state = RuleState.APPLIED
        return r
