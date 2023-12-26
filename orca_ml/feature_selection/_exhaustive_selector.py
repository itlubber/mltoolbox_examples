import operator
from functools import reduce
from itertools import chain, combinations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import sem
from scipy.stats._continuous_distns import t
from sklearn.metrics import check_scoring
from sklearn.model_selection._validation import cross_val_score, _score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ._base import SelectorMixin
from ..base import BaseEstimator, MetaEstimatorMixin, clone


def _calc_score(estimator, X, y, indices, groups=None, scoring=None, cv=None, **fit_params):
    _, n_features = X.shape
    mask = np.in1d(np.arange(n_features), indices)
    X = X[:, mask]
    if cv is None:
        # scorer = check_scoring(estimator, scoring=scoring)
        try:
            estimator.fit(X, y, **fit_params)
        except Exception:
            scores = np.nan
        else:
            scores = _score(estimator, X, y, scoring)
        scores = np.asarray([scores], dtype=np.float64)
    else:
        scores = cross_val_score(estimator, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=None,
                                 pre_dispatch='2*n_jobs', error_score=np.nan, fit_params=fit_params)
    return mask, scores


def ncr(n, r):
    """Return the number of combinations of length r from n items.

    Parameters
    -----------
    n : int
        Total number of items
    r : int
        Number of items to select from n

    Returns
    -----------
    Number of combinations, integer
    """
    r = min(r, n - r)
    if r == 0:
        return 1
    numerator = reduce(operator.mul, range(n, n - r, -1))
    denominator = reduce(operator.mul, range(1, r + 1))
    return numerator // denominator


class ExhaustiveFeatureSelector(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Exhaustive Feature Selection for Classification and Regression.

    Parameters
    -----------
    estimator : scikit-learn classifier or regressor

    min_features : int (default: 1)
        Minimum number of features to select

    max_features : int (default: 1)
        Maximum number of features to select

    verbose : bool (default: True)
        Prints progress as the number of epochs to stdout.

    scoring : str, (default='_passthrough_scorer')
        Scoring metric in faccuracy, f1, precision, recall, roc_auc) for classifiers,
        {'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2'} for regressors,
        or a callable object or function with signature ``scorer(estimator, X, y)``.

    cv : int (default: 5)
        Scikit-learn cross-validation generator or `int`,
        If estimator is a classifier (or y consists of integer class labels), stratified k-fold is performed, and regular k-fold cross-validation otherwise.
        No cross-validation if cv is None, False, or 0.

    n_jobs : int (default: 1)
        The number of CPUs to use for evaluating different feature subsets in parallel. -1 means 'all CPUs'.

    pre_dispatch : int, or string (default: '2*n_jobs')
        Controls the number of jobs that get dispatched during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
        Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
            None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
            An int, giving the exact number of total jobs that are spawned
            A string, giving an expression as a function of n_jobs, as in `2*n_jobs

    Attributes
    -----------
    subset_info_ : list of dicts
        A list of dictionary with the following keys:
            'support_mask', mask array of the selected features
            'cv_scores', cross validate scores

    support_mask_ : array-like of booleans
        Array of final chosen features

    best_idx_ : array-like, shape = [n_predictions]
        Feature Indices of the selected feature subsets.
    best_score_ : float
        Cross validation average score of the selected subset.
    best_feature_indices_ : array-like, shape = (n_features,)
        Feature indices of the selected feature subsets.

    Examples
    -----------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from orca_ml.feature_selection.exhaustive_feature_selector import ExhaustiveFeatureSelector
    >>> X, y = load_iris(return_X_y=True, as_frame=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> efs = ExhaustiveFeatureSelector(knn, min_features=1, max_features=4, cv=3)
    >>> efs.fit(X, y)
    ExhaustiveFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3), max_features=4)
    >>> efs.best_score_
    0.9733333333333333
    >>> efs.best_idx_
    12
    """
    def __init__(self, estimator, *, min_features=1, max_features=1, scoring=None, cv=3, verbose=0, n_jobs=None, pre_dispatch='2*n_jobs'):
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch

    def _validate_params(self, X, y):
        X, y = check_X_y(X, y, estimator=self.estimator)
        _, n_features = X.shape
        if not isinstance(self.min_features, int) or (self.max_features > n_features or self.max_features < 1):
            raise AttributeError("max_features must be smaller than %d and larger than 0" % (n_features + 1))
        if not isinstance(self.min_features, int) or (self.min_features > n_features or self.min_features < 1):
            raise AttributeError("min_features must be smaller than %d and larger than 0" % (n_features + 1))
        
        if self.max_features < self.min_features:
            raise AttributeError("min_features must be less equal than max_features")
        return X, y

    def fit(self, X, y, groups=None, **fit_params):
        """Perform feature selection and learn model from training data.

        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples, )
            Target values.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set. Passed to the fit method of the cross-validator.
        fit_params : dict
            Parameters to pass to the fit method of classifier

        Returns
        -----------
        self : ExhaustiveFeatureSelector
        """
        X, y = self._validate_params(X, y)
        _, n_features = X.shape
        min_features, max_features = self.min_features, self.max_features
        candidates = chain.from_iterable(combinations(range(n_features), r=i) for i in range(min_features, max_features + 1))
        # chain has no __len__ method
        n_combinations = sum(ncr(n=n_features, r=i) for i in range(min_features, max_features + 1))

        estimator = self.estimator
        scoring = check_scoring(estimator, self.scoring)
        cv = self.cv
        n_jobs = self.n_jobs
        pre_dispatch = self.pre_dispatch
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        work = enumerate(parallel(delayed(_calc_score)(clone(estimator), X, y, c, groups=groups, scoring=scoring, cv=cv, **fit_params) for c in candidates))
        
        subset_info = []
        append_subset_info = subset_info.append
        try:
            for iteration, (mask, cv_scores) in work:
                avg_score = np.nanmean(cv_scores).item()
                append_subset_info({"support_mask": mask, "cv_scores": cv_scores, "avg_score": avg_score})
                if self.verbose:
                    print("Feature set: %d/%d, avg score: %.3f" % (iteration + 1, n_combinations, avg_score))
        except KeyboardInterrupt:
            print("Stopping early due to keyboard interrupt...")
        finally:
            max_score = float("-inf")
            best_idx, best_info = -1, {}
            for i, info in enumerate(subset_info):
                if info["avg_score"] > max_score:
                    max_score = info["avg_score"]
                    best_idx, best_info = i, info
            score = max_score
            mask = best_info["support_mask"]
            self.subset_info_ = subset_info
            self.support_mask_ = mask
            self.best_idx_ = best_idx
            self.best_score_ = score
            self.best_feature_indices_ = np.where(mask)[0]
            return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_
    
    def get_metrics(self, confidence_interval=0.95):
        """Return metric dictionary

        Parameters
        -----------
        confidence_interval : float, default=0.95
            A positive float between 0.0 and 1.0 to compute the confidence interval bounds of the CV score averages.

        Returns
        -----------
        d : dict
            Dictionary with items where each dictionary value is a list with the number of iterations (number of feature subsets) as
            its length. The dictionary keys corresponding to these lists are as follows:
                'support_mask': boolean array indicates the selected features
                'cv_scores': list with individual CV scores
                'avg_scores': average CV score
                'std_dev': standard deviation of the CV score average
                'std_err': standard error of the CV score average
                'ci_bound': confidence interval bound of the average CV score
        """
        check_is_fitted(self, "subset_info_")
        subset_info = self.subset_info_
        l = []
        for i, info in enumerate(subset_info):
            std_dev = np.std(info["cv_scores"])
            bound, std_err = _calc_confidence(info["cv_scores"], confidence=confidence_interval)
            extra_info = {"ci_bound": bound, "std_dev": std_dev, "std_err": std_err}
            l.append({**info, **extra_info})
        return l


def _calc_confidence(scores, confidence=0.95):
    std_err = sem(scores)
    bound = std_err * t._ppf((1 + confidence) / 2.0, len(scores))
    return bound, std_err
