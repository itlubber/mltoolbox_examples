from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..base import ModelParameterProxy


class HistGradientBoostingClassifierParameterProxy(ModelParameterProxy):
    def __init__(self, loss='auto', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0., max_bins=255, monotonic_cst=None, warm_start=False,
                 early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-7,
                 verbose=0, random_state=None):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.monotonic_cst = monotonic_cst
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.n_iter_no_change = n_iter_no_change

    def _make_estimator(self):
        self.estimator = HistGradientBoostingClassifier(
            loss=self.loss, learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
            max_bins=self.max_bins, monotonic_cst=self.monotonic_cst, warm_start=self.warm_start,
            early_stopping=self.early_stopping, scoring=self.scoring, validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change, tol=self.tol, verbose=self.verbose, random_state=self.random_state,
        )


class HistGradientBoostingRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, loss='least_squares', *, learning_rate=0.1,
                 max_iter=100, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0., max_bins=255,
                 monotonic_cst=None, warm_start=False, early_stopping='auto',
                 scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-7,
                 verbose=0, random_state=None):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.monotonic_cst = monotonic_cst
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _make_estimator(self):
        self.estimator = HistGradientBoostingRegressor(
        loss=self.loss, learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
            max_bins=self.max_bins, monotonic_cst=self.monotonic_cst, warm_start=self.warm_start,
            early_stopping=self.early_stopping, scoring=self.scoring, validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change, tol=self.tol, verbose=self.verbose, random_state=self.random_state,
        )
