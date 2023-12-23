import os
from pathlib import Path
import joblib
import datetime
from sklearn. model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import catboost
from catboost import CatBoostClassifier, Pool

from ._search_space import get_parameters
from ._optimizer import get_param_optimizer


# TODO:树棵树配置
class SPOTBase:
        
    def __init__(self, estimator, thresholds, tree_nums=None, param_names=None, optimizer_name="HyperOPT", scoring=None, 
                 config=None, save_path="model", max_iter=50):
        self.estimator = estimator
        self.thresholds = thresholds
        self.tree_nums = tree_nums
        self.param_names = param_names
        self.optimizer_name = optimizer_name
        self.save_path = save_path
        self.scoring = scoring
        self.config = config
        self.max_iter = max_iter
        self.best_estimator = None

    def check_params(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def update_init_model(self, class_name, train_params):
        if class_name == "LGBMClassifier":
            train_params["init_model"] = self.best_estimator
        elif class_name == "CatBoostclassifier":
            train_params["init_model"] = self.best_estimator
        elif class_name == "XGBClassifier":
            train_params[ "xgb_model"] = self.best_estimator
        return train_params

    def fit(self, X, y):
        self.check_params()
        optimizer = get_param_optimizer(self.optimizer_name, self.estimator, X, y)
        class_name = self.estimator.__class__.__name__
        params = self.config if self.config is not None else get_parameters(class_name, self.param_names)
        filename = ""
        train_params = {"params": params, "num_boost_round": 200, "early_stopping_rounds": 50}

        if isinstance(self.estimator, (XGBClassifier, LGBMClassifier)):
            train_params["verbose_eval"] = False

        for i, thr in enumerate(self.thresholds):
            print(f"\ncurrent threshold is {thr}.")
            train_params["params"] = params
            if self.tree_nums is not None:
                for tree_num in self.tree_nums:
                    train_params["num_boost_round"] = tree_num
                    print(f"\tcurrent tree amount is {tree_num}.")
                    self.best_estimator = optimizer.staged_optimize(thr, train_params, self.max_iter, self.scoring)
                    if self.best_estimator is not None:
                        train_params = self.update_init_model(class_name, train_params)
            else:
                self.best_estimator = optimizer.staged_optimize(thr, train_params, self.max_iter, self.scoring)
                if self.best_estimator is not None:
                    train_params = self.update_init_model(class_name, train_params)
        
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            filename = Path(self.save_path) / f"{self.optimizer_name}_{class_name}_{train_params['num_boost_round']}_{str(cur_time)}.pkl"
            self.export_model(filename, self.best_estimator)
        
        print(f"best estimator path: {filename}")
        return self

    def predict(self, X):
        if self.best_estimator is None:
            raise ValueError("not found a best estimator")
        return self.best_estimator.predict(X)

    def export_model(self, filename, estimator):
        joblib.dump(estimator, filename=filename)

    def load_model(self, filename):
        return joblib.load(filename=filename)


class SPOTClassifier(SPOTBase):
    def __init__(self, estimator, thresholds, tree_nums=None, param_names=None, optimizer_name="HyperOPT", scoring=None, config=None, save_path="model", max_iter=50):
        super().__init__(estimator, thresholds, tree_nums, param_names, optimizer_name, scoring, config, save_path, max_iter)
