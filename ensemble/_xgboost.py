from collections import namedtuple
import numpy as np
from xgboost.sklearn import XGBClassifier, XGBRegressor, XGBRanker, XGBRFClassifier, XGBRFRegressor

from ..base import ModelParameterProxy


def has_xgboost():
    try:
        import xgboost
        return True
    except ImportError:
        return False


def isa_xgboost_model(clf):
    if not has_xgboost():
        return False
    from xgboost.sklearn import XGBModel # noqa: F401
    return isinstance(clf, XGBModel)


def is_xgboost_model(cls):
    if not has_xgboost():
        return False
    from xgboost.sklearn import XGBModel # noqa: F401
    return issubclass(cls, XGBModel)


Node = namedtuple('Node', ['feature', 'node_id', 'left_id', 'right_id', 'missing_id', 'leaf_val', 'split_val', 'is_leaf'])
# Tree = namedtuple('Tree', ['node', ‘left', 'right'])


class Tree:
    def __init__(self, node: Node, left: 'Tree' = None, right: 'Tree' = None):
        self.node = node
        self.left = left
        self.right = right


# from dataclasses import dataclass
# @dataclass
# class Node(object):
#     feature: str = None
#     node_id: int = None
#     left_id: int = None
#     right_id: int = None
#     missing_id: int = None
#     leaf_val: float = None
#     split_val: float = None
#     is_leaf: bool = False


# @dataclass
# class Tree(object):
#     node: Node
#     left: 'Tree' = None
#     right: 'Tree' = None


class XGBClassifierParameterProxy(ModelParameterProxy):
    def __init__(self, objective="binary:logistic", max_depth=None, learning_rate=None, n_estimators=100,
        verbosity=None, booster=None, tree_method=None, n_jobs=None, gamma=None,
        min_child_weight=None, max_delta_step=None, subsample=None,
        colsample_bytree=None, colsample_bylevel=None, colsample_bynode=None, reg_alpha=None, reg_lambda=None,
        scale_pos_weight=None, base_score=None, random_state=None, missing=np.nan, num_parallel_tree=None,
        monotone_constraints=None, interaction_constraints=None, importance_type="gain", gpu_id=None, validate_parameters=None):
        self.objective = objective
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id =gpu_id
        self.validate_parameters = validate_parameters

    def _make_estimator(self):
        self.estimator = XGBClassifier(
            objective=self.objective,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbosity=self.verbosity,
            booster=self.booster,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            random_state=self.random_state,
            missing=self.missing,
            num_parallel_tree=self.num_parallel_tree,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters,
        )


class XGBRFClassifierParameterProxy(ModelParameterProxy):
    def __init__(self, objective="binary:logistic", max_depth=None, learning_rate=1, n_estimators=100, verbosity=None, booster=None,
                 tree_method=None, n_jobs=None, gamma=None, min_child_weight=None, max_delta_step=None, subsample=0.8,
                 colsample_bytree=None, colsample_bylevel=None, colsample_bynode=0.8, reg_alpha=None, reg_lambda=1e-5,
                 scale_pos_weight=None, base_score=None, random_state=None, missing=np.nan, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None, importance_type="gain", gpu_id=None, validate_parameters=None):
        self.objective = objective
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters

    def _make_estimator(self):
        self.estimator = XGBRFClassifier(
            objective=self.objective,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbosity=self.verbosity,
            booster=self.booster,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            missing=self.missing,
            num_parallel_tree=self.num_parallel_tree,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters
        )


class XGBRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, objective="reg:squarederror", max_depth=None, learning_rate=None, n_estimators=100, verbosity=None, booster=None,
                 tree_method=None, n_jobs=None, gamma=None, min_child_weight=None, max_delta_step=None, subsample=None,
                 colsample_bytree=None, colsample_bylevel=None, colsample_bynode=None, reg_alpha=None, reg_lambda=None,
                 scale_pos_weight=None, base_score=None, random_state=None, missing=np.nan, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None, importance_type="gain", gpu_id=None, validate_parameters=None):
        self.objective = objective
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters

    def _make_estimator(self):
        self.estimator = XGBRegressor(
            objective=self.objective,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbosity=self.verbosity,
            booster=self.booster,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            random_state=self.random_state,
            missing=self.missing,
            num_parallel_tree=self.num_parallel_tree,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters
        )


class XGBRFRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, objective="reg:squarederror", max_depth=None, learning_rate=1, n_estimators=100, verbosity=None, booster=None,
                 tree_method=None, n_jobs=None, gamma=None, min_child_weight=None, max_delta_step=None, subsample=0.8,
                 colsample_bytree=None, colsample_bylevel=None, scale_pos_weight=None, base_score=None, random_state=None,
                 colsample_bynode=0.8, reg_alpha=None, reg_lambda=1e-5, missing=np.nan, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None, importance_type="gain", gpu_id=None, validate_parameters=None):
        self.objective = objective
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters

    def _make_estimator(self):
        self.estimator = XGBRFRegressor(
            objective=self.objective,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbosity=self.verbosity,
            booster=self.booster,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            random_state=self.random_state,
            missing=self.missing,
            num_parallel_tree=self.num_parallel_tree,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters
        )


class XGBRankerParameterProxy(ModelParameterProxy):
    def __init__(self, objective="rank:pairwise", max_depth=None, learning_rate=None, n_estimators=100, verbosity=None, booster=None,
                 tree_method=None, n_jobs=None, gamma=None, min_child_weight=None, max_delta_step=None, subsample=None,
                 colsample_bytree=None, colsample_bylevel=None, colsample_bynode=None, reg_alpha=None, reg_lambda=None,
                 scale_pos_weight=None, base_score=None, random_state=None, missing=np.nan, num_parallel_tree=None,
                 monotone_constraints=None, interaction_constraints=None, importance_type="gain", gpu_id=None, validate_parameters=None):
        self.objective = objective
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.validate_parameters = validate_parameters

    def _make_estimator(self):
        self.estimator = XGBRanker(
            objective=self.objective,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            verbosity=self.verbosity,
            booster=self.booster,
            tree_method=self.tree_method,
            n_jobs=self.n_jobs,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            base_score=self.base_score,
            random_state=self.random_state,
            missing=self.missing,
            num_parallel_tree=self.num_parallel_tree,
            monotone_constraints=self.monotone_constraints,
            interaction_constraints=self.interaction_constraints,
            importance_type=self.importance_type,
            gpu_id=self.gpu_id,
            validate_parameters=self.validate_parameters,
        )


estimator_params = ["learning_rate", "n_estimators", "reg_alpha", "reg_lambda", "random_state", "n_jobs", "eval_set", "verbose"]
booster_params = ["eta", "num_boost_round", "alpha", "lambda", "seed", "nthread", "evals", "verbose_eval"]


def convert_estimator_params(params):
    # params = estimator.get_xgb_params()
    res = {k: v for k, v in params.items()}
    for old_k, new_k in zip(estimator_params, booster_params):
        res[new_k] = res.pop(old_k)
    return res


def convert_booster_params(params):
    res = {k: v for k, v in params.items()}
    for old_k, new_k in zip(booster_params, estimator_params):
        res[new_k] = res.pop(old_k)
    return res
