import numpy as np
from sklearn. feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
from .._base import SelectorMixin
from ...base import BaseEstimator


class VifSelector(BaseEstimator, SelectorMixin):
    """Filter features according to its variance inflation factor.

    Parameters
    ----------
    threshold : float or str, default=None
        The vif threshold for feature selection
    n_jobs : int, default=None
        Number of parallel computing.

    Examples
    ----------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> from mltoolbox.feature_selection.collinear._vif import VifSelector
    >>> X, y = make_classification(n_ samples=200, n_features=10, random_state=1)
    >>> x = pd.DataFrame(x)
    >>> selector = VifSelector(threshold-10).fit(x)
    >> selector.scores_
    array([1.06258447, inf, 1.0220527, inf, inf, 1.00886036, inf, 1.04588531, 1.0182847 , 1.01107023])
    >>> selector.transform(X)
    [200 rows x 6 columns]
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=10, n_jobs=None):
        self.threshold = threshold
        self.n_jobs = n_jobs

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        threshold = _calculate_threshold(self, scores, self.threshold)
        return scores <= threshold

    def fit(self, X, y=None, **fit_params):
        self.scores_ = vif(X, y, n_jobs=self.n_jobs)
        return self

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
        }


def vif(X, y=None, n_jobs=None):
    # set min features to 2!
    X = check_array(X, accept_sparse=False, dtype="numeric", ensure_2d=True, ensure_min_features=2, force_all_finite=True)
    _, n_features = X.shape
    scores = Parallel(n_jobs=n_jobs)(delayed(variance_inflation_factor)(X, i) for i in range(n_features))
    return np.asarray(scores, dtype=np.float64)
    # return np.fromiter((variance_inflation_factor(X, i) for i in range (n_features)), dtype=np.float64)


def vif_scores(X, y=None, n_jobs=None):
    selector = VifSelector(n_jobs=n_jobs)
    selector.fit(X, y=y)
    return selector.scores_
