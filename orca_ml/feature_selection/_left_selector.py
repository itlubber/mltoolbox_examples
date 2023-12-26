import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_is_fitted, check_array, column_or_1d, _deprecate_positional_args
from sklearn.preprocessing import LabelEncoder
from ._base import SelectorMixin
from ..base import BaseEstimator


def _lift_score(y_true, y_pred):
    """Calculate lift according to label data.

    Parameters
    -----------
    y_true : array-like
    y_pred : array-like

    Returns
    -----------
    lift : float

    Examples
    -----------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1])
    >>> _lift_score(y_true, y_pred) # (5 / 7) / (6 / 9)
    1.0714285714285716
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    base_bad_rate = np.average(y_true)
    hit_bad_rate = np.count_nonzero((y_true == 1) & (y_pred == 1)) / np.count_nonzero(y_pred)
    return hit_bad_rate / base_bad_rate


class LiftSelector(BaseEstimator, SelectorMixin):
    """Feature selection via lift score.

    Parameters
    -----------
    threshold : float or str (default=3.0)
        Feature which has a lift score greater than `threshold` will be kept.
    n_jobs : int or None, (default=None)
        Number of parallel.
    
    Attributes
    -----------
    threshold_: float
        The threshold value used for feature selection.
    scores_ : array-like of shape (n_features,)
        Lift scores of features.
    
    Examples
    -----------
    >>> from orca_ml.feature_selection import LiftSelector
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=3.0, n_jobs=None):
        self.threshold = threshold
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        _, n_features= X.shape
        self.n_features_in_ = n_features
        self.scores_ = _lift_scores(X, y, n_jobs=self.n_jobs)
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        threshold = _calculate_threshold(self, scores, self.threshold)
        return scores > threshold
    
    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)
    
    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_y": True,
        }


def _lift_scores(X, y, *, n_jobs=None):
    X = check_array(X, dtype="numeric", force_all_finite=True, ensure_2d=True)
    if not np.all(np.isin(X, [0, 1])):
        raise ValueError("X should only contain 0 and 1!")
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(le.classes_) != 2:
        raise ValueError("Only support binary label for computing lift!")
    _, n_features = X.shape
    scores = Parallel(n_jobs=n_jobs)(delayed(_lift_score)(y, X[:, i]) for i in range(n_features))
    return np.asarray(scores, dtype=np.float64)


def lift_scores(X, y, *, n_jobs=None):
    selector = LiftSelector(n_jobs=n_jobs)
    selector.fit(X, y)
    return selector.scores_
