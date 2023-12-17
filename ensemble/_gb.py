from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from ..base import ModelParameterProxy


class GradientBoostingClassifierParameterProxy(ModelParameterProxy):
    def _init_(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.,
                max_depth=3, min_impurity_decrease=0.,
                min_impurity_split=None, init=None,
                random_state=None, max_features=None, verbose=0,
                max_leaf_nodes=None, warm_start=False,
                validation_fraction=0.1,
                n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        
    def _make_estimator(self):
        estimator = GradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            init=self.init,
            min_impurity_split=self.min_impurity_split,
            random_state=self.random_state,
            verbose=self.verbose,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            ccp_alpha=self.ccp_alpha,
        )
        self.estimator = estimator


class GradientBoostingRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.,
                max_depth=3, min_impurity_decrease=0.,
                min_impurity_split=None, init=None, random_state=None,
                max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                validation_fraction=0.1, warm_start=False,
                n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha

    def _get_estimator_class(self):
        estimator = GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            criterion=self.criterion,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            init=self.init,
            random_state=self.random_state,
            max_features=self.max_features,
            alpha=self.alpha,
            verbose=self.verbose,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            ccp_alpha=self.ccp_alpha,
        )
        self.estimator = estimator
