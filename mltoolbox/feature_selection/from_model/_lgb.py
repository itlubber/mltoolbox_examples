from sklearn.utils.validation import _deprecate_positional_args

from ._base import BaseSelectFromModel
from ...ensemble._lightgbm import LGBMClassifierParameterProxy, LGBMRegressorParameterProxy


class SelectFromLGBMClassifier(LGBMClassifierParameterProxy, BaseSelectFromModel):
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1, top_k=None, importance_getter='auto',
                boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100,
                subsample_for_bin=200000, objective=None, class_weight="balanced",
                min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                subsample=1., subsample_freq=0, colsample_bytree=1.,
                reg_alpha=0., reg_lambda=0., random_state=None,
                n_jobs=None, silent=True, importance_type='split', is_unbalance=False):
        LGBMClassifierParameterProxy.__init__(self,
                                                boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth,
                                                learning_rate=learning_rate, n_estimators=n_estimators,
                                                subsample_for_bin=subsample_for_bin, objective=objective,
                                                class_weight=class_weight,
                                                min_split_gain=min_split_gain, min_child_weight=min_child_weight,
                                                min_child_samples=min_child_samples,
                                                subsample=subsample, subsample_freq=subsample_freq,
                                                colsample_bytree=colsample_bytree,
                                                reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=random_state,
                                                n_jobs=n_jobs, silent=silent,
                                                importance_type=importance_type, is_unbalance=is_unbalance)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)


class SelectFromLGBMRegressor(LGBMRegressorParameterProxy, BaseSelectFromModel):
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1, top_k=None, importance_getter='auto',
                boosting_type='gbdt', num_leaves=31, max_depth=-1,
                subsample_for_bin=200000, objective=None, class_weight="balanced",
                min_split_gain=0., min_child_weight=1e-3, min_child_samples=20,
                subsample=1., subsample_freq=0, colsample_bytree=1.,
                reg_alpha=0., reg_lambda=0., random_state=None,
                learning_rate=0.1, n_estimators=100,
                n_jobs=-1, silent=True, importance_type='split'):
        LGBMRegressorParameterProxy.__init__(self,
                                            boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth,
                                            learning_rate=learning_rate, n_estimators=n_estimators,
                                            subsample_for_bin=subsample_for_bin, objective=objective,
                                            class_weight=class_weight,
                                            min_split_gain=min_split_gain, min_child_weight=min_child_weight,
                                            min_child_samples=min_child_samples,
                                            subsample=subsample, subsample_freq=subsample_freq,
                                            colsample_bytree=colsample_bytree,
                                            reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=random_state,
                                            n_jobs=n_jobs, silent=silent, importance_type=importance_type)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)
