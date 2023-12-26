from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_is_fitted, check_X_y, _deprecate_positional_args
from statsmodels.api import Logit

from ._base import SelectorMixin
from ..base import BaseEstimator


class PValueSelector(BaseEstimator, SelectorMixin):
    """Select feature by logit pvalue scores.

    Examples
    ----------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> from mltoolbox.feature_selection. _pval_selector import PValueSelector
    >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=0)l
    >>> X, y = pd.DataFrame(X), pd.Series(y)
    >>> selector = PValueSelector(threshold=0.02).fit(X, y)
    >>> selector.transform(X)
    >>> selector.get_selected_features(X)
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=0.001, start_params=None, method='newton', maxiter=35, full_output=0, disp=1, callback=None):
        self.threshold = threshold
        self.start_params = start_params
        self.method = method
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback

    def _get_support_mask(self):
        check_is_fitted(self, "pvalues_")
        pvalues = self.pvalues_
        threshold = _calculate_threshold(self, pvalues, self.threshold)
        return pvalues <= threshold

    @property
    def threshold_(self):
        check_is_fitted(self, "pvalues_")
        return _calculate_threshold(self, self.pvalues_, self.threshold)

    def fit(self, X, y=None, **fit_params):
        self.tvalues_, self.pvalues_ = get_logit_result(X, y)
        return self

    def _more_tags(self):
        return {"X_types": ["2darray"], "allow_nan": False, "requires_y": True}


def get_logit_result(X, y, *, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None):
    X, y = check_X_y(X, y, dtype="numeric", ensure_2d=True, force_all_finite=True, y_numeric=True)
    binary_result = Logit(y, X).fit(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback)
    tvalues = binary_result.tvalues
    pvalues = binary_result.pvalues
    return tvalues, pvalues


def pvalue_scores(X, y, *, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None):
    """p-value scores of the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples, n_classes)
    start_params
    method
    maxiter
    full_output
    disp
    callback

    Returns
    ----------
    pvalues : array-like of shape (n_features,)
        p-values in logit result
    """
    selector = PValueSelector(start_params=start_params, method=method, maxiter=maxiter, full_output=full_output, disp=disp, callback=callback)
    selector.fit(X, y)
    return selector.pvalues_
