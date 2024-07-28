import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing._label import _unique
from sklearn.utils.validation import check_is_fitted, check_array, _deprecate_positional_args

from ._base import SelectorMixin
from ..base import BaseEstimator, MetaEstimatorMixin
from ..preprocessing.discretization._base import BaseDiscretizer


class InformationValueSelector(BaseEstimator, SelectorMixin):
    """Feature selection via information value.

    Parameters
    ----------
    threshold : float or str (default=0.02)
        Feature which has a information value greater than threshold will be kept.
    regularization : int or float (default=1.0)
        Regularization item that adds up to information value calculation.
    n_jobs : int or None, (default=None)
        Number of parallel.

    Attributes
    ----------
    threshold_ : float
        The threshold value used for feature selection.
    scores_ : array-like of shape (n_features,)
        Information values of features.

    Notes
    ----------
    Only allows discretized data

    Examples
    ----------
    >>> from mltoolbox.datasets import load_uci_credit
    >>> from mltoolbox.feature_selection import InformationValueSelector
    >>> x, y = load_uci_credit(return_X_y=True, as_frame=True)
    >>> x
        [30000 rows x 23 columns]
    >>> selector = InformationValueSelector().fit(x, y)
    >>> selector.transform(x)
        [30000 rows x 21 columns]
    >>> selector.get_selected_features(x)
    array(['LIMIT_BAL', 'EDUCATION', 'AGE', 'PAY_?', 'PAY_2', 'PAY_3',
            'PAY_4', 'PAY_5','PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], dtype=object)
    """

    @_deprecate_positional_args
    def __init__(self, *, threshold=0.02, regularization=1.0, n_jobs=None):
        self.threshold = threshold
        self.regularization = regularization
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        self.n_features_in_ = X.shape[1]
        self.scores_ = _iv(X, y, regularization=self.regularization, n_jobs=self.n_jobs)
        self.feature_names_in_ = self.get_selected_features(X)
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self. threshold)
        return self.scores_ > threshold
    
    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)
    
    def _more_tags(self):
        return {
            "X_types": ["categorical"],
            "allow_nan": False,
            "requires_y": True,
        }


class BinningInformationValueSelector(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    @_deprecate_positional_args
    def __init__(self, estimator, *, threshold=0.02, regularization=1.0, n_jobs=None):
        self.estimator = estimator
        self.threshold = threshold
        self.regularization = regularization
        self.n_jobs = n_jobs
    
    def fit(self, X, y=None, **fit_params):
        estimator = self.estimator
        if not isinstance(estimator, BaseDiscretizer):
            raise TypeError("estimator should be a discretizer.")
        self.n_features_in_ = X.shape[1]
        Xt = estimator.fit_transform(X, y)
        self.scores_ = _iv(Xt, y, regularization=self.regularization, n_jobs=self.n_jobs)
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


IVSelector = InformationValueSelector
BinningIVSelector = BinningInformationValueSelector


def _iv(X, y, regularization=1.0, n_jobs=None):
    """Parallel compute the information value of each feature.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Discretized data
    y : array-like of shape (n_samples,)
        Binary classification label.
    regularization : float, must be greater than 0
        Regularization item adds to the information value calculation.
    n_jobs : int or None, default is None.
        Number of parallel.

    Returns
    ----------
    iv_values : array-like of shape (n_features,)
    """
    X = check_array(X, dtype=None, force_all_finite=True, ensure_2d=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(le.classes_) != 2:
        raise ValueError("Only support binary label for computing information value!")
    _, n_features = X.shape
    iv_values = Parallel(n_jobs=n_jobs)(delayed(_col_iv)(X[:, i], y, regularization=regularization) for i in range(n_features))
    return np.asarray(iv_values, dtype=np.float64)


def _gen_iv_value(x, y, regularization=1.0):
    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization
    for cat in _unique(x):
        mask = x == cat
        # Ignore unique values. This helps to prevent overfitting on id-Like columns.
        if np.count_nonzero(mask) == 1:
            yield 0.
        else:
            n_event = np.count_nonzero(np.logical_and(mask, event_mask)) + regularization
            n_nonevent = np.count_nonzero(np.logical_and(mask, nonevent_mask)) + regularization
            event_rate = n_event / event_tot
            nonevent_rate = n_nonevent / nonevent_tot
            woe = np.log(event_rate / nonevent_rate)
            iv = (event_rate - nonevent_rate) * woe
            yield iv


def _col_iv(x, y, regularization=1.0):
    return sum(v for v in _gen_iv_value(x, y, regularization=regularization))


def col_iv(x, y, regularization=1.0):
    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization
    uniques = _unique(x)
    n_cats = len(uniques)
    event_rates = np.zeros(n_cats, dtype=np.float64)
    nonevent_rates = np.zeros(n_cats, dtype=np.float64)
    for i, cat in enumerate(uniques):
        mask = x == cat
        event_rates[i] = np.count_nonzero(mask & event_mask) + regularization
        nonevent_rates[i] = np.count_nonzero(mask & nonevent_mask) + regularization

    # Ignore unique values. This helps to prevent overfitting on id-like columns.
    bad_pos = (event_rates + nonevent_rates) == (2 * regularization + 1)
    event_rates /= event_tot
    nonevent_rates /= nonevent_tot
    ivs = (event_rates - nonevent_rates) * np.log(event_rates / nonevent_rates)
    ivs[bad_pos] = 0.
    return np.sum(ivs).item()


def iv_scores(X, y, *, regularization=1.0, n_jobs=None):
    selector = InformationValueSelector(regularization=regularization, n_jobs=n_jobs)
    selector.fit(X, y)
    return selector.scores_


def binning_iv_scores(estimator, X, y, *, regularization=1.0, n_jobs=None):
    selector = BinningInformationValueSelector(estimator, regularization-regularization, n_jobs=n_jobs)
    selector.fit(X, y)
    return selector.scores_
