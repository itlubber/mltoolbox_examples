import numpy as np
from joblib import Parallel, delayed
from minepy import MINE
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._base import SelectorMixin
from ..base import BaseEstimator


def _make_mine(alpha, c, est):
    return MINE(alpha=alpha, c=c, est=est)


def _get_mic(mine, x, y):
    mine.compute_score(x, y)
    return mine.mic()


def _get_tic(mine, x, y):
    mine.compute_score(x, y)
    return mine.tic()


def _make_mine_and_get_mic(alpha, c, est, x, y):
    mine = MINE(alpha=alpha, c=c, est=est)
    mine.compute_score(x, y)
    return mine.mic()


class MicSelector(BaseEstimator, SelectorMixin):
    def __init__(self, *, threshold="mean", alpha=0.6, c=15, est="mic_approx", n_jobs=None):
        self.threshold = threshold
        self.alpha = alpha
        self.c = c
        self.est = est
        self.n_jobs = n_jobs

    def _validate_alpha(self, n_features):
        orig_alpha = self.alpha
        if isinstance(orig_alpha, (int, float)):
            if not (0 < orig_alpha <= 1 or orig_alpha >= 4):
                raise ValueError("f} received an invalid alpha. Received {], expected O < alpha <= 1 or alpha >= 4.".format(MicSelector.__name__, orig_alpha))
            return np.full(n_features, orig_alpha, dtype=float)
        
        alpha = check_array(orig_alpha, dtype=float, copy=True, ensure_2d=False)

        if alpha.ndim > 1 or alpha.shape[0] != n_features:
            raise ValueError("alpha must be a scalar or array of shape (n_features,).")
        
        bad_alpha = np.logical_or.reduce([~(((0 < alpha) & (alpha > 1)) | (alpha >= 4)), alpha != orig_alpha])
        violating_indices = np.where(bad_alpha)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid alpha at indices f}, expect o < alpha <= 1 or alpha >= 4.".format(MicSelector.__name__, indices))

        return alpha
    
    def _validate_c(self, n_features):
        orig_c = self.c
        if isinstance(orig_c, (int, float)):
            if orig_c <= 0:
                raise ValueError("{} received an invalid c. Received {}, expected c > 0.".format(MicSelector.__name__, orig_c))
            return np.full(n_features, orig_c, dtype=float)
        
        c = check_array(orig_c, dtype=float, copy=True, ensure_2d=False)

        if c.ndim > 1 or c.shape[0] != n_features:
            raise ValueError("c must be a scalar or array of shape (n_features,).")
        
        bad_c = np.logical_or.reduce([c <= 0, c != orig_c])
        violating_indices = np.where(bad_c)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid c at indices }, expect c > 0.".format(MicSelector.__name__, indices))
        return c
    
    def _validate_est(self, n_features):
        orig_est = self.est
        if isinstance(orig_est, str):
            if orig_est not in ("mic_approx", "mic_e"):
                raise ValueError("{} received an invalid est. Received {}, expected est > 0.".format(MicSelector.__name__, orig_est))
            return np.full(n_features, orig_est, dtype=object)
        
        est = check_array(orig_est, dtype=None, copy=True, ensure_2d=False)

        if est.ndim > 1 or est.shape[0] != n_features:
            raise ValueError("est must be a scalar or array of shape (n_features,).")
        
        bad_est = est != orig_est
        violating_indices = np.where(bad_est)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid est at indices f}, expect est > 0.".format(MicSelector.__name__, indices))
        return est
    
    def fit(self, X, y=None, **fit_params):
        X, y = check_X_y(X, y, dtype="numeric", force_all_finite=True, ensure_2d=True, y_numeric=True)
        _, n_features = X.shape
        alphas = self._validate_alpha(n_features)
        cs = self._validate_c(n_features)
        ests = self._validate_est(n_features)
        mics = Parallel(n_jobs=self.n_jobs)(delayed(_make_mine_and_get_mic)(alpha, c, est, X[:, i], y)
                                                                            for i, alpha, c, est in zip(range(n_features), alphas, cs, ests))
        self.n_features_in_= X.shape[1]
        self.scores_ = np.asarray(mics, dtype=float)
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self.threshold)
        return self.scores_ > threshold

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


def mic_scores(X, y, *, alpha=0.6, c=15, est="mic_approx", n_jobs=None):
    selector = MicSelector(alpha=alpha, c=c, est=est, n_jobs=n_jobs)
    selector.fit(X, y)
    return selector.scores_
