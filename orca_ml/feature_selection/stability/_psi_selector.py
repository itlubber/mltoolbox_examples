import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.model_selection import check_cv
from sklearn.utils import _safe_indexing
from sklearn.utils._encode import _unique
from sklearn.utils.validation import check_is_fitted, indexable, check_array, _deprecate_positional_args
from .._base import SelectorMixin
from ...base import BaseEstimator, MetaEstimatorMixin
from ...preprocessing.discretization import BaseDiscretizer


_PSI_PCT_EPSILON = 1e-3


class PSISelector(BaseEstimator, SelectorMixin):
    """Population Stability Index"""

    @_deprecate_positional_args
    def __init__(self, *, threshold=0.001, cv=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        self.threshold = threshold
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self.threshold)
        return self.scores_ <= threshold

    def fit(self, X, y=None, groups=None):
        X, groups = indexable(X, groups)
        X = check_array(X, dtype=None, ensure_2d=True, force_all_finite=True)
        uniques = [_unique(X[:, i]) for i in range(X.shape[1])]
        cv = check_cv(self.cv)
        n_jobs = self.n_jobs
        verbose = self.verbose
        pre_dispatch = self.pre_dispatch

        cv_scores = []
        for train, test in cv.split(X, y, groups):
            X_train = _safe_indexing(X, train)
            X_test = _safe_indexing(X, test)
            scores = _psi_score(X_train, X_test, uniques, n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
            cv_scores.append(scores)
        self.scores_ = np.mean(cv_scores, axis=0)
        return self

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        return _calculate_threshold(self, scores, self.threshold)

    def _more_tags(self):
        return {
            "X_types": ["categorical"],
            "allow_nan": False,
        }


class BinningPSISelector (BaseEstimator, SelectorMixin, MetaEstimatorMixin):
    @_deprecate_positional_args
    def __init__(self, estimator, *, threshold=0.001, cv=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.threshold = threshold
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self.threshold)
        return self.scores_ <= threshold

    def fit(self, X, y=None, groups=None):
        estimator = self.estimator
        if not isinstance(estimator, BaseDiscretizer):
            raise TypeError("estimator should be a discretizer.")
        X = estimator.fit_transform(X, y)
        X, groups = indexable(X, groups)
        X = check_array(X, dtype=None, ensure_2d=True, force_all_finite=True)
        uniques = [_unique(X[:, i]) for i in range(X.shape[1])]
        cv = check_cv(self.cv)
        n_jobs = self.n_jobs
        verbose = self.verbose
        pre_dispatch = self.pre_dispatch
        cv_scores = []
        for train, test in cv.split(X, y, groups):
            X_train = _safe_indexing(X, train)
            X_test = _safe_indexing(X, test)
            scores = _psi_score(X_train, X_test, uniques, n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
            cv_scores.append(scores)
        self.scores_ = np.mean(cv_scores, axis=0)
        return self
    
    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        scores = self.scores_
        return _calculate_threshold(self, scores, self.threshold)
    
    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
        }


def _psi_score(train, test, uniques, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_single_psi_score)(train[:, i], test[:,i], unique) for i, unique in enumerate(uniques))
    return scores


def _single_psi_score(expected, actual, unique_values):
    n_expected = len(expected)
    n_actual = len(actual)
    psi_index = []
    for value in unique_values:
        expected_cnt = np.count_nonzero(expected == value)
        actual_cnt = np.count_nonzero(actual == value)
        expected_cnt = expected_cnt if expected_cnt else 1.
        actual_cnt = actual_cnt if actual_cnt else 1.
        expected_rate = expected_cnt / n_expected
        actual_rate = actual_cnt / n_actual
        psi_index.append((actual_rate - expected_rate) * np.log(actual_rate / expected_rate))
    return sum(psi_index)


def compute_psi(X, y=None, *, estimator=None, groups=None, cv=None, n_jobs=None, pre_dispatch='2*n_jobs', verbose=0):
    if estimator:
        if not isinstance(estimator, BaseDiscretizer):
            raise TypeError("estimator should be a discretizer.")
        X = estimator.fit_transform(X, y)
    X, groups = indexable(X, groups)
    X = check_array(X, dtype=None, ensure_2d=True, force_all_finite=True)
    uniques = [_unique(X[:, i]) for i in range(X.shape[1])]
    cv = check_cv(cv)
    cv_scores = []
    for train, test in cv.split(X, y, groups):
        X_train = _safe_indexing(X, train)
        X_test = _safe_indexing(X, test)
        scores = _psi_score(X_train, X_test, uniques, n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
        cv_scores.append(scores)
    scores_ = np.mean(cv_scores, axis=0)
    return scores_


def calc_psi(train, test, n_bins=20, bin_type='quantile'):
    """Calculate psi value.

    :param train: expected distribution array
    :param test: actual distribution array
    :param n_bins: bucket number, default is 20
    :param bin_type: bucket type, default is 'quantile', can be 'uniform'
    :return: psi value by specific bucketing type

    :example:
    >>> import pandas as pd
    >>> data - pd.DataFrame({('y1': [2, 3, 1, 4, 5, 2, 0, 18, 2, 6, 3, 5], 'y2': [1, 2, 4, 2, 2, 6, 5, 3, 0, 5, 4, 18]})
    >>> y1,y2 = data['y1'], data['y2']
    >>> calc_psi(y1, y2, bin_type='quantile')
    0.11552453009332421
    >>> calc_psi(y1, y2, bin_type='origin')
    0.11552453009332421
    """
    if bin_type == 'uniform':
        m, M = np.min(train), np.max(train)
        breakpoints = np.linspace(m, M, n_bins + 1)
    elif bin_type == 'quantile':
        breakpoints = np.quantile(train, np.linspace(0, 1, n_bins + 1))
        # breakpoints = np.percentile(train, np.linspace(0, 100, n_bins + 1))
    
    # DEPRECATED: guess not correct
    elif bin_type == 'origin':
        sorted_score = np.unique(train)
        n = len(sorted_score) - 1
        # ids = np.fromiter((round(i / n_bins * n) for i in range (n_bins + 1)), dtype=np.int)
        ids = np.fromiter((round((i + 1) / (n_bins + 1) * n) for i in range(n_bins + 1)), dtype=int)
        breakpoints = sorted_score[ids]
    else:
        raise ValueError("Unknown bucket type {}!".format(repr(bin_type)))
    psi = sum((train_pct - test_pct) * np.log(train_pct / test_pct) for train_pct, test_pct in generate_psi_pairs(train, test, breakpoints))
    return psi


def generate_psi_pairs(expected, actual, breakpoints):
    """Generate pairs for psi calculation

    :param expected: expected distribution array
    :param actual: actual distribution array
    :param breakpoints: break points array
    :return: yield out train test percent pairs

    :example:
    >>> import pandas as pd
    >>> data - pd.DataFrame({'y1': [2, 3, 1, 4, 5, 2, 0, 18, 2, 6, 3, 5], 'y2': [1, 2, 4, 2, 2, 6, 5, 3, 0, 5, 4, 18]})
    >>> y1, y2 = data['y1'], data['y2']
    >>> breakpoints = np.array([0, 6, 12, 18])
    >>> for train_pct, test_pct in generate_psi_pairs(y1, y2, breakpoints):
            print(train_pct, test_pct)
    0.8333333333333334  0.8333333333333334
    0.001               0.001
    0.08333333333333333 0.08333333333333333
    """
    expected_num, actual_num = len(expected), len(actual)
    for s1, s2 in zip(breakpoints[:-1], breakpoints[1:]):
        train_pct = np.count_nonzero((s1 < expected) & (expected <= s2)) / expected_num
        test_pct = np.count_nonzero((s1 < actual) & (actual <= s2)) / actual_num
        train_pct = train_pct if train_pct else train_pct + _PSI_PCT_EPSILON
        test_pct = test_pct if test_pct else test_pct + _PSI_PCT_EPSILON
        yield train_pct, test_pct


def date_split(X, test_size=None, train_size=None, date_col="user_date"):
    """Split dataframe or series into train and test subsets by datetime quantiles.

    :param X: whole data
    :param test_size: time quantile of the train dataset
    :param train_size: time quantile of the test dataset
    :param date col: date columr
    :return: List containing train-test split of inputs
    """
    if test_size is not None and train_size is not None:
        raise ValueError("Only one can be chosen from train_size and test_size'!")
    elif test_size is not None:
        q = 1.0 - test_size
    elif train_size is not None:
        q = train_size
    else:
        # if test_size is None and train size is None:
        raise ValueError("Only one can be chosen from train_size and test_size!")
    
    date = X[date_col]
    time_splitpoint = date.quantile(q)
    before_pos = (date <= time_splitpoint)
    after_pos = (date > time_splitpoint)
    X_before, X_after = X[before_pos], X[after_pos]
    return X_before, X_after
