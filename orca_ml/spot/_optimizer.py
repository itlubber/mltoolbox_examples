import copy
import warnings
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from flaml import AutoML, tune
from flaml.model import BaseEstimator
from hyperopt import Trials, fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from sklearn.metrics import roc_auc_score
from sklearn. model_selection import train_test_split


warnings.filterwarnings("ignore")


class ParameterOptimizer:
    name = None

    def __init__(self):
        if self.name is None:
            raise NotImplementedError(f"name is not defined")
    
    def get_module(self, estimator):
        if isinstance(estimator, LGBMClassifier):
            return lgb
        elif isinstance(estimator, CatBoostClassifier):
            return cat
        elif isinstance(estimator, XGBClassifier):
            return xgb
        return None


class HyperOPTParameterOptimizer(ParameterOptimizer):
    name = "HyperOPT"

    def __init__(self, estimator, X, y, random_state=None):
        super().__init__()
        self.estimator = estimator
        self.class_name = estimator.__class__.__name__
        self.train_params = None
        self.random_state = random_state
        self.init_train_data(estimator, X, y)

    def init_train_data(self, estimator, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_state)
        if isinstance(estimator, LGBMClassifier):
            train_data, test_data = lgb.Dataset(X_train, y_train, free_raw_data=False), lgb.Dataset(X_test, y_test, free_raw_data=False)
            train_x, test_x = X_train, X_test
        elif isinstance(estimator, CatBoostClassifier):
            train_data, test_data = Pool(X_train, y_train), Pool(X_test, y_test)
            train_x, test_x = X_train, X_test
        elif isinstance(estimator, XGBClassifier):
            train_data, test_data = xgb.DMatrix(X_train, y_train), xgb.DMatrix(X_test, y_test)
            train_x, test_x = train_data, test_data
        else:
            raise TypeError(f"{estimator} is not a valid estimator.")
        
        self.train_data = train_data
        self.test_data = test_data

        self.train_x = train_x
        self.test_x = test_x

        self.train_label = y_train
        self.test_label = y_test

    def _translate_params(self, params):
        param = {}
        for k, v in params.items():
            if isinstance(v, list):
                param[k] = hp.choice(k, v)
            else:
                param[k] = v
        return param

    def calculate_score(self, scorer, estimator):
        train_preds, test_preds = estimator.predict(self.train_x), estimator.predict(self.test_x)
        train_score, test_score = scorer(self.train_label, train_preds), scorer(self.test_label, test_preds)
        return train_score, test_score
    
    def update_eval_data(self, params):
        if isinstance(self.estimator, LGBMClassifier):
            params["train_set"] = self.train_data
            params["valid_sets"] = [self.train_data, self.test_data]
        elif isinstance(self.estimator, CatBoostClassifier):
            params[ "dtrain"] = self.train_data
            params[ "evals"] = self.test_data
        elif isinstance(self.estimator, XGBClassifier):
            params["dtrain"] = self.train_data
            params["evals"] = [(self.train_data, "Train"), (self.test_data, "Valid")]
        else:
            raise TypeError(f"{self.estimator} is not a valid estimator.")

    def _object_fn(self, space):
        module, thr, scorer = space['module'], space['threshold'], space['scorer']
        train_params = {k: v for k, v in self.train_params.items()} # Pool is not deepcору
        train_params["params"] = space['hp_params']
        model = module.train(**train_params)

        train_score, test_score = self.calculate_score(scorer, model)
        status = STATUS_OK if train_score <= thr else STATUS_FAIL
        return {"loss": -test_score, "status": status}

    def staged_optimize(self, thr, train_params, max_iter=50, scoring=None):
        params = train_params["params"]
        self.update_eval_data(train_params)
        self.train_params = train_params
        scoring = scoring if callable(scoring) else roc_auc_score
        hp_params = self._translate_params(params)
        module = self.get_module(self.estimator)
        space = {"threshold": thr, "hp_params": hp_params, "module": module, "scorer": scoring}
        try:
            cur_best_params = fmin(fn=self._object_fn, space=space, algo=tpe.suggest, max_evals=max_iter, trials=Trials())
        except Exception as e:
            print("current not found a batter model.")
            return None
        best_params = {param_name: params[param_name][ind] for param_name, ind in cur_best_params.items()}
        best_params.update({k: v for k, v in params.items() if not isinstance(v, list)})
        print(f"train params is {best_params}")

        train_params["params"] = best_params
        model = module.train(**train_params)
        train_score, test_score = self.calculate_score(scoring, model)
        print(f"train score is: {train_score}, test score is: {test_score}")
        return model


class FlamlParameterOptimizer(ParameterOptimizer):
    name = "Flaml"

    # TODO:自定义目标函数实现
    def __init__(self, estimator, X, y, random_state=None):
        super().__init__()
        self.estimator = estimator
        self.class_name = estimator.__class__.__name__
        self.X = X
        self.y = y
        self.random_state = random_state
    
    def _create_class(self, class_name, params):
        base_class = BaseEstimator

        @classmethod
        def search_space(cls, data_size, task):
            space = {}
            for name, vals in params.items():
                if isinstance(vals, list):
                    space[name] = {"domain": tune.choice(list(vals))}
            return space
        
        def logregobj():
            pass

        def init(cls, **config):
            base_class.__init__(cls, **config)
            cls.estimator_class = self.estimator.__class__
    
        attrs_and_methods = {"__init__": init, "search_space": search_space}
        # attrs_and_methods.update(**set_params)
        est_class = type(class_name, (base_class,), attrs_and_methods)

        return est_class
    
    def calculate_score(self, scorer, estimator, X, y):
        return scorer(estimator.predict_proba(X), y)
    
    def staged_optimize(self, thr, train_params, max_iter=50, scoring=None):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=self.random_state)
        automl = AutoML()
        class_name = "SPOT" + self.estimator.__class__.__name__
        new_class = self._create_class(class_name, train_params["params"])
        
        automl.add_learner(class_name, new_class)
        automl.fit(X_train=X_train, y_train=y_train, task="classification", max_iter=max_iter, estimator_list=[class_name])
        if callable(scoring):
            train_score, test_score = self.calculate_score(scoring, automl, X_train, y_train), self.calculate_score(scoring, automl, X_test, y_test)
        else:
            train_score, test_score = automl.score(X_train, y_train), automl.score(X_test, y_test)
        print(f"train score is: {train_score}, test score is: {test_score}")
        return automl.model.model


def get_param_optimizer(name, estimator, X, y):
    if name == "HyperOPT":
        return HyperOPTParameterOptimizer(estimator, X, y)
    elif name == "Flaml":
        return FlamlParameterOptimizer(estimator, X, y)
