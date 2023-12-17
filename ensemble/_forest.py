from pandas import DataFrame
from sklearn.ensemble._forest import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import RandomTreesEmbedding as SkRandomTreesEmbedding

from ..base import ModelParameterProxy


class ForestClassifierParameterProxy(ModelParameterProxy):
    def _init_(self, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def _get_estimator_class(self):
        raise NotImplementedError

    def _make_estimator(self):
        estimator_class = self._get_estimator_class()
        estimator = estimator_class(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            min_samples_split=self.min_samples_split,
        )
        self.estimator = estimator


class ForestRegressorParameterProxy(ModelParameterProxy):
    def __init__(self,
                n_estimators=100,
                criterion="mse",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
                ):
            self.n_estimators = n_estimators
            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.min_impurity_split = min_impurity_split
            self.bootstrap = bootstrap
            self.oob_score = oob_score
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.verbose = verbose
            self.warm_start = warm_start
            self.ccp_alpha = ccp_alpha
            self.max_samples = max_samples

    def _get_estimator_class(self):
        raise NotImplementedError
    
    def _make_estimator(self):
        estimator_class = self._get_estimator_class()
        estimator = estimator_class(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )
        self.estimator = estimator


class RandomForestClassifierParameterProxy(ForestClassifierParameterProxy):
    def _get_estimator_class(self):
        return RandomForestClassifier


class RandomForestRegressorParameterProxy(ForestRegressorParameterProxy):
    def _get_estimator_class(self):
        return RandomForestRegressor


class ExtraTreesClassifierParameterProxy(ForestClassifierParameterProxy):
    def _get_estimator_class(self):
        return ExtraTreesClassifier


class ExtraTreesRegressorParameterProxy(ForestRegressorParameterProxy):
    def _get_estimator_class(self):
        return ExtraTreesRegressor


class RandomTreesEmbedding(SkRandomTreesEmbedding):
    def transform(self, X):
        data = super(RandomTreesEmbedding, self).transform(X)
        if isinstance(X, DataFrame):
            index = X.index
            columns = ["random_trees_embedding%d" % (i + 1) for i in range(data.shape[1])]
            return DataFrame(data=data.toarray(), columns=columns, index=index)
        return data


class RandomTreesEmbeddingParameterProxy(ModelParameterProxy):
    def _init_(self, n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0., max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None,
    sparse_output=False, n_jobs=None, random_state=None, verbose=0, warm_start=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.sparse_output = sparse_output
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.warm_start = warm_start

    def _make_estimator(self):
        estimator = RandomTreesEmbedding(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            sparse_output=self.sparse_output,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start
        )
        self.estimator = estimator
