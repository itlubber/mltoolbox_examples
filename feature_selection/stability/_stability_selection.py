"""Stability selection

This module contains a scikit-learn compatible implementation of stability selection [1]_.

References
-----------
.. [1] Meinshausen, N. and Buhlmann, P., 2010. Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), pp.417-473.
.. [2] Shah, R.D. and Sanworth, R.J., 2013. Variable selection with error control: another look at stability selection. Journal of the Rayal Statistical Society: Series B (Statistical Methodology), 75(1), pp.55-80.
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection._from_model import _get_feature_importances, _calculate_threshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y, check_random_state, safe_mask
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args

from ._bootstrap import bootstrap_without_replacement, complementary_pairs_bootstrap, stratified_bootstrap
from .._base import SelectorMixin
from ...base import BaseEstimator, clone


BOOTSTRAP_FUNC_MAPPING = {
    'subsample': bootstrap_without_replacement,
    'complementary_pairs': complementary_pairs_bootstrap,
    'stratified': stratified_bootstrap
}


def _return_estimator_from_pipeline(pipeline):
    if isinstance(pipeline, Pipeline):
        return pipeline._final_estimator
    else:
        return pipeline


def _bootstrap_generator(n_bootstrap_iterations, bootstrap_func, y, n_subsamples, random_state=None):
    for _ in range(n_bootstrap_iterations):
        subsamples = bootstrap_func(y, n_subsamples, random_state=random_state)
        if isinstance(subsamples, tuple):
            yield from subsamples
        else:
            yield subsamples


def _fit_bootstrap_sample(base_estimator, importance_getter, X, y, lambda_name, lambda_value, threshold=None):
    """
    Fit base estimator on a bootstrap sample of the original data, and return a mas of the variables that are selected by the fitted model.

    Parameters
    -----------
    base_estimator : sklearn.base.BaseEstimator
    X : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,)
        The target values.
    lambda_name : str
        Name of the penalization parameter of base estimator.
    lambda_value : float
        Value of the penalization parameter of base estimator.
    threshold: string or float, default=None
        The threshold value to use for feature selection.
        Feature whose importance is greater or equal are kept while the others are discarded. If "median" (resp. "mean") then the threshold value
        is the median (resp. the mean) of the feature importances. A scaling factor (e.g., "1.25 * mean") may also be used. If None and if the estimator
        has a parameter penalty set to 11, either explicitly or implicitly (e.g., Lasso), the threshold used is 1e-5. Otherwise, "mean" is used by default.
    
    Returns
    -----------
    selected_variables : array-like of shape (n_features,)
        Boolean mask of the selected features.
    """
    base_estimator.set_params(**{lambda_name: lambda_value})
    base_estimator.fit(X, y)
    estimator = _return_estimator_from_pipeline(base_estimator)
    scores = _get_feature_importances(estimator, importance_getter, transform_func='norm', norm_order=1)
    threshold = _calculate_threshold(estimator, scores, threshold)
    mask = np.full(scores.shape, True, dtype=bool)
    mask[scores < threshold] = False
    return mask


def plot_stability_path(stability_selection, **kwargs):
    """Plot stability path.

    Parameters
    -----------
    stability_selection : StabilitySelection
    kwargs : dict
        Arguments passed to matplotlib plot function.
    """
    check_is_fitted(stability_selection, "stability_scores_")
    stability_scores = stability_selection.stability_scores_
    threshold = stability_selection.threshold
    paths_to_highlight = stability_selection._get_support_mask()

    lambda_grid = stability_selection.lambda_grid
    x_grid = lambda_grid / np.max(lambda_grid)

    fig, ax = plt.subplots(1, 1, **kwargs)
    if not paths_to_highlight.all():
        ax.plot(x_grid, stability_scores[~paths_to_highlight].T, 'k:', linewidth=0.5)

    if paths_to_highlight.any():
        ax.plot(x_grid, stability_scores[paths_to_highlight].T, 'r-', linewidth=0.5)

    ax.plot(x_grid, threshold * np.ones_like(lambda_grid), 'b--', linewidth=0.5)
    ax.set_ylabel('Stability score')
    ax.set_xlabel('Lambda / max(Lambda)')

    fig.tight_layout()
    return fig, ax


class StabilitySelection(BaseEstimator, SelectorMixin):
    """
    Stability selection [1]_ fits the estimator `base_estimator` on bootstrap samples of the original data set, for different values of
    the regularization parameter for `base_estimator`. Variables that reliably get selected by the model in these bootstrap samples are
    considered to be stable variables.

    Parameters
    -----------
    base_estimator : sklearn.base.BaseEstimator
        The base estimator used for stability selection. The estimator must have either a `feature_importances_` or `coef_` attribute after fitting.
    
    lambda_name : str
        The name of the penalization parameter for the estimator `base_estimator`
    
    lambda_grid : array-like
        Grid of values of the penalization parameter to iterate over.
    
    n_bootstrap_iterations : int
        Number of bootstrap samples to create.
    
    sample_fraction : float, default=None
        The fraction of samples to be used in each bootstrap sample. Should be between 0 and 1. If 1, all samples are used.
    
    threshold : float
        Threshold defining the minimum cutoff value for the stability scores.
    
    bootstrap_func : str or callable fun (default=bootstrap_without_replacement)
        The function used to subsample the data. This parameter can be:
            - A string, which must be one of
                'subsample`: For subsampling without replacement.
                'complementary_pairs': For complementary pairs subsampling [2]_.
                'stratified': For stratified bootstrapping in imbalanced classification.
            - A function that takes y, and a random state
                as inputs and returns a list of sample indices in the range
                (0, (len(y) - 1)). By default, indices are uniformly subsampled.
    
    bootstrap_threshold : string or float, default="mean"
        The threshold value to use for feature selection.
        Feature whose importance is greater or equal are kept while the others are discarded. If "median" (resp. "mean") then the ``threshold`` value
        is the median (resp. the mean) of the feature importances. A scaling factor (e.g., "1.25 * mean") may also be used. If None and if the estimator
        has a parameter penalty set to 11, either explicitly or implicitly (e.g., Lasso), the threshold used is 1e-5. Otherwise, "mean" is used by default.
    
    verbose : int
        Controls the verbosity: the higher, the more messages.
    
    n_jobs : int, default=1
        Number of jobs to run in parallel
   
    pre_dispatch : int or string, default=None
        Controls the number of jobs that get dispatched during parallel execution. Reducing this number can be useful to avoid an explosion
        of memory consumption when more jobs get dispatched than CPUs can process.
        This parameter can be:
            - None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays to on-demand spawning of the jobs
            - An int, giving the exact number of total iobs that are spawned.
            - A string, giving an expression as a function of n_jobs, as in '2*n_jobs'
    
    random_state : int, np.random.RandomState or None, default=None
        Pseudo random number generator state used for random uniform sampling from lists of possible values instead of scipy.stats distributions.
        If int, random_state is the seed used by the random number generator;
        If np.random.RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np random`.

    Attributes
    -----------
    stability_scores_ : array-like of shape (n_features, n_alphas)
        Array of stability scores for each feature for each value of the penalization parameter.

    References
    -----------
    .. [1] Meinshausen, N. and Buhlmann, P., 2010. Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4), pp.417-473.
    .. [2] Shah, R.D. and Samworth, R.J., 2013. Variable selection with error control: another look at stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 75(1), pp.55-80.
    """
    @_deprecate_positional_args
    def __init__(self, base_estimator=LogisticRegression(), 
                 importance_getter='auto', 
                 lambda_name='C',
                 lambda_grid=np.logspace(-5, -2, 25),
                 n_bootstrap_iterations=20, sample_fraction=0.5, threshold=0.6,
                 bootstrap_func=bootstrap_without_replacement,
                 bootstrap_threshold=None, verbose=0, n_jobs=None, pre_dispatch='2*n_jobs', random_state=None):
        self.base_estimator = base_estimator
        self.importance_getter = importance_getter
        self.lambda_name = lambda_name
        self.lambda_grid = lambda_grid
        self.n_bootstrap_iterations = n_bootstrap_iterations
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.bootstrap_func = bootstrap_func
        self.bootstrap_threshold = bootstrap_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state

    def _validate_input(self):
        if not isinstance(self.n_bootstrap_iterations, int) or self.n_bootstrap_iterations <= 1.0:
            raise ValueError('n_bootstrap_iterations should be a positive integer, got %s' % self.n_bootstrap_iterations)
        if not isinstance(self.sample_fraction, float) or not (0.0 < self.sample_fraction <= 1.0):
            raise ValueError('sample_fraction should be a float in (0, 1], got %s' % self.sample_fraction)
        if not isinstance(self.threshold, float) or not (0.0 < self.threshold <= 1.0):
            raise ValueError('threshold should be a float in (0, 1], got %s' % self.threshold)
        if self.lambda_name not in self.base_estimator.get_params():
            raise ValueError('lambda_name is set to %s, but base_estimator %s does not have a parameter with that name' % (
                self.lambda_name, self.base_estimator.__class__.__name__))
        if isinstance(self.bootstrap_func, str):
            if self.bootstrap_func not in BOOTSTRAP_FUNC_MAPPING:
                raise ValueError('bootstrap_func is set to %s, but must be one of %s or a callable' % (self.bootstrap_func, BOOTSTRAP_FUNC_MAPPING.keys()))
            
            self.bootstrap_func = BOOTSTRAP_FUNC_MAPPING[self.bootstrap_func]
        elif not callable(self.bootstrap_func):
            raise ValueError('bootstrap_func must be one of %s or a callable' % BOOTSTRAP_FUNC_MAPPING.keys())
    
    def fit(self, X, y):
        """Fit the stability selection model on the given data.

        Parameters
        -----------
        X : farray-like, sparse matrix}, shape = [n_samples, n features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        self._validate_input()

        X, y = check_X_y(X, y, accept_sparse='csr')

        n_samples, n_features = X.shape
        n_subsamples = np.floor(self.sample_fraction * n_samples).astype(int)
        n_lambdas = self.lambda_grid.shape[0]

        base_estimator = clone(self.base_estimator)
        random_state = check_random_state(self.random_state)
        stability_scores = np.zeros((n_features, n_lambdas))

        for idx, lambda_value in enumerate(self.lambda_grid):
            if self.verbose > 0:
                print("Fitting estimator for lambda = %.5f (%d / %d) on %d bootstrap samples" % (lambda_value, idx + 1, n_lambdas, self.n_bootstrap_iterations))
            
            bootstrap_samples = _bootstrap_generator(self.n_bootstrap_iterations, self.bootstrap_func, y, n_subsamples, random_state=random_state)
            selected_variables = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)(delayed(_fit_bootstrap_sample)(
                clone(base_estimator),
                self.importance_getter,
                X=X[safe_mask(X, subsample), :],
                y=y[subsample],
                lambda_name=self.lambda_name,
                lambda_value=lambda_value,
                threshold=self.bootstrap_threshold) for subsample in bootstrap_samples)
            
            stability_scores[:, idx] = np.vstack(selected_variables).mean(axis=0)
        
        self.stability_scores_ = stability_scores
        self.scores_ = stability_scores.max(axis=1)
        return self

    def _get_support_mask(self):
        """Get a mask, or integer index, of the features selected

        Returns
        -----------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape [# input features], in which an element is True iff its corresponding feature is selected for retention. 
            If `indices` is True, this is an integer array of shape [# output features] whose values are indices into the input feature vector.
        """
        check_is_fitted(self, "scores_")
        return self.scores_ > self.threshold
