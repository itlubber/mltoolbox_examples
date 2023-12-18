from sklearn.utils.validation import _deprecate_positional_args

from ._base import BaseSelectFromModel
from ...ensemble._xgboost import XGBClassifierParameterProxy, XGBRegressorParameterProxy


class SelectFromXGBClassifier(XGBClassifierParameterProxy, BaseSelectFromModel):
    @_deprecate_positional_args
    def __init__(
                    self, *, threshold=None, norm_order=1, top_k=None, importance_getter='auto',
                    max_depth=3, learning_rate=0.1, n_estimators=100,
                    objective="binary:logistic",booster='gbtree',
                    n_jobs=None, gamma=0, min_child_weight=1, max_delta_step=0,
                    subsample=1, colsample_bytree=1, colsample_bylevel=1,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                    base_score=0.5, random_state=None, missing=None, importance_type="gain"
                ):
        XGBClassifierParameterProxy.__init__(self,
                                            max_depth=max_depth, learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            objective=objective, booster=booster,
                                            n_jobs=n_jobs, gamma=gamma, min_child_weight=min_child_weight,
                                            max_delta_step=max_delta_step,
                                            subsample=subsample, colsample_bytree=colsample_bytree,
                                            colsample_bylevel=colsample_bylevel,
                                            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                            scale_pos_weight=scale_pos_weight,
                                            base_score=base_score, random_state=random_state, missing=missing,
                                            importance_type=importance_type)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)


class SelectFromXGBRegressor(XGBRegressorParameterProxy, BaseSelectFromModel):
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1, top_k=None,
                importance_getter='auto',
                max_depth=3, learning_rate=0.1, n_estimators=100,
                objective="binary:logistic", booster='gbtree',
                n_jobs=None, gamma=0, min_child_weight=1, max_delta_step=0,
                subsample=1, colsample_bytree=1, colsample_bylevel=1,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                base_score=0.5, random_state=0, missing=None, importance_type="gain",
                ):
        XGBRegressorParameterProxy.__init__(self,
                                            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                                            objective=objective, booster=booster,
                                            n_jobs=n_jobs, gamma=gamma, min_child_weight=min_child_weight,
                                            max_delta_step=max_delta_step,
                                            subsample=subsample, colsample_bytree=colsample_bytree,
                                            colsample_bylevel=colsample_bylevel,
                                            reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                            scale_pos_weight=scale_pos_weight,
                                            base_score=base_score, random_state=random_state, missing=missing,
                                            importance_type=importance_type)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)
