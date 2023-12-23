param_config = {
    "XGBClassifier": {
        # "n_estimators”: [80, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.],
        "min_child_weight": list(range(1, 21)),
        "max_delta_step": [0, 1],
        "subsample": [i * 0.01 for i in range(70, 95, 3)],
        "colsample_bytree": [i *0.01 for i in range(70, 95, 3)],
        "reg_alpha": list(range(0, 15)),
        "reg_1ambda": list(range(0, 15)),
        "eval_metric": "auc",
        'objective': 'binary:logistic',
        'random_state': 42,
    },
    "LGBMClassifier": {
        # "n_estimators": [80, 100, 120, 140, 160, 180, 200],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.2, 0.1, 0.05, 0.01, 0.001],
        "reg_alpha": [0, 1, 5, 10, 70, 100, 150],
        "bagging_fraction": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "feature_fraction": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "scale_pos_weight": [10, 15, 30, 60],
        "max_bin": [10, 30, 50, 100, 200, 300, 500, 1000, 1500],
        "min_gain_to_split": [0, 5, 10, 30, 50],
        "bagging_freq": [3, 5, 10, 20],
        "num_leaves": [30, 60, 90, 100, 200, 300, 500, 1000, 1200, 1500],
        "verbose": -1,
        'task': 'train',
        'metric': {'auc'},
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'random_state': 42,
    },
    "CatBoostClassifier": {
        'iterations': [80, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500],
        'learning_rate': [0.2, 0.1, 0.05, 0.01, 0.001],
        '12_leaf_reg': list(range(1, 10)),
        'bagging_temperature': [i * 0.01 for i in range(1, 100)],
        'depth': list(range(1, 11)),
        'one_hot_max_size': [9, 12, 15],
        'eval_metric': "AUC",
        "verbose": False,
    },
}


def get_parameters(class_name, param_names):
    param_dict = param_config.get(class_name)
    if param_names is None or len(param_names) == 0:
        return param_dict
    return {name: param_dict[name] for name in param_names}
