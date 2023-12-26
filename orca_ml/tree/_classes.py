from sklearn.tree._classes import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from ..base import ModelParameterProxy


class BaseDecisionTreeParameterProxy(ModelParameterProxy):
    def __init__(self, criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None,
                random_state=None, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight="balanced", ccp_alpha=0.):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split= min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def _get_estimator_class(self):
        return BaseDecisionTree

    def _make_estimator(self):
        estimator_class = self._get_estimator_class()
        estimator = estimator_class(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            min_impurity_split=self.min_impurity_split,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
        )
        self.estimator = estimator


class DecisionTreeClassifierParameterProxy(BaseDecisionTreeParameterProxy):
    def _get_estimator_class(self):
        return DecisionTreeClassifier


class DecisionTreeRegressorParameterProxy(BaseDecisionTreeParameterProxy):
    def _get_estimator_class(self):
        return DecisionTreeRegressor


class ExtraTreeClassifierParameterProxy(BaseDecisionTreeParameterProxy):
    def _get_estimator_class(self):
        return ExtraTreeClassifier


class ExtraTreeRegressorParameterProxy(BaseDecisionTreeParameterProxy):
    def _get_estimator_class(self):
        return ExtraTreeRegressor
