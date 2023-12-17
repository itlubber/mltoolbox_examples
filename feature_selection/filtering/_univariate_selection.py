import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect as SKGenericUnivariateSelect
from sklearn.feature_selection import SelectFdr as SKSelectFdr
from sklearn.feature_selection import SelectFpr as SKSelectFpr
from sklearn.feature_selection import SelectFwe as SKSelectFwe
from sklearn.feature_selection import SelectKBest as SKSelectKBest
from sklearn.feature_selection import SelectPercentile as SKSelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_selection._univariate_selection import _chisquare

from .._base import SelectorMixin
from ...base import BaseEstimator


score_func_map_for_classification = {
    "chi2": chi2,
    "f_classif": f_classif,
    "mutual_info": mutual_info_classif,
}

score_func_map_for_regression = {
    "f_regression": f_regression,
    "mutual_info": mutual_info_regression,
}


def _get_score_func(y, score_func):
    """
    Parameters
    -----------
    y : array-like
    score_func : str or callable

    Returns
    -----------
    score_func : callable
    """
    target_type = type_of_target(y)
    if target_type in ("continuous", "continuous-multioutput"):
        score_func_map = score_func_map_for_regression
    elif target_type in ("binary", "multiclass", "multiclass-multioutput", "multilabel-indicator"):
        score_func_map = score_func_map_for_classification
    else:
        raise ValueError("Unknown target type!")
    if isinstance(score_func, str):
        score_func = score_func_map.get(score_func, None)
    if score_func is None:
        raise ValueError("Score function name not found.")
    return score_func


class _BaseFilter(SelectorMixin, BaseEstimator):
    """Initialize the univariate feature_selection.

    Parameters
    -----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.
    """
    def __init__(self, score_func):
        self.score_func = score_func

    def fit(self, X, y):
        """Run score function on (X, y) and get the appropriate features.
        
        Parameters
        -----------
        self : instance of ``_BaseFilter``
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -----------
        self
        """
        self.score_func = _get_score_func(y, self.score_func)
        # super().fit(X, y)
        X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)
        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) was passed." % (self.score_func, type(self.score_func)))
        self._check_params(X, y)
        score_func_ret = self.score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None
        
        self.scores_ = np.asarray(self.scores_)
        return self
    
    def _check_params(self, X, y):
        pass

    def _more_tags(self):
        return {'requires_y': True}


class GenericUnivariateSelect(SKGenericUnivariateSelect, SelectorMixin):
    pass


class SelectFdr(SKSelectFdr, SelectorMixin):
    pass


class SelectFwe(SKSelectFwe, SelectorMixin):
    pass


class SelectFpr(SKSelectFpr, SelectorMixin):
    pass


class SelectKBest(SKSelectKBest, SelectorMixin):
    pass


class SelectPercentile(SKSelectPercentile, SelectorMixin):
    pass
