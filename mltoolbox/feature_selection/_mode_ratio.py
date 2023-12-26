import numpy as np
from joblib import Parallel, delayed
from pandas import Series
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args
from ._base import SelectorMixin
from .. base import BaseEstimator


class ModeFilter(BaseEstimator, SelectorMixin):
    """Filter features according to its column mode ratio

    If the feature's mode ratio is greater than threshold, it will be filtered.

    Parameters
    -----------
    threshold : float or str, default=0.95
        The mode ratio's threshold
    n_jobs : int, default=None

    Examples
    -----------
    >>> import pandas as pd
    >>> from orca ml.feature_selection import ModeFilter
    >>> X = pd.DataFrame([[0, np.nan, 0, 3], [0, np.nan, 4, 3], [0, 1, np.nan, 3]], columns=["f1", "f2", "f3", "f4"])
    >>> selector = ModeFilter(threshold="mean”).fit(X)
    >> selector.transform(X)
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=0.95, n_jobs=None):
        self.threshold = threshold
        self.n_jobs = n_jobs
    
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        threshold = _calculate_threshold(self, scores, self.threshold)
        return scores <= threshold
    
    def fit(self, X, y=None, **fit_params):
        X = check_array(X, dtype=None, ensure_2d=True, force_all_finite="allow-nan")
        _, n_features = X.shape
        scores = Parallel(n_jobs=self.n_jobs)(delayed(_col_mode_ratio)(X[:, i]) for i in range(n_features))
        # scores = np.fromiter((_col_mode_ratio(X[:, i]) for i in range(n_features)), dtype=пp.float64)
        self.scores_ = np.asarray(scores, dtype=np.float64)
        return self

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)

    # aсcept data with missing values
    def _more_tags(self):
        return {
            "X_types": ["2darray", "categorical", "string"],
            "allow_nan": True,
        }


def _col_mode_ratio(col):
    col = Series(col)
    summary = col.value_counts(dropna=True)
    return summary.iloc[0] / sum(summary) if len(summary) > 0 else 1.0
    # _, counts = stats.mode(X, axis=0, nan_policy='omit')
    # imputer_mask = _get_mask(X, np.nan)
    # return counts[0] / np.sum(~imputer_mask, axis=0)


def mode_ratios(X, y=None, n_jobs=None):
    """
    Parameters
    -----------
    X: array-like
    y : array-like
    n_jobs : int, default=None
    """
    selector = ModeFilter(n_jobs=n_jobs)
    selector.fit(X, y)
    return selector.scores_

