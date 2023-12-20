import numpy as np
from pandas import DataFrame
from ..base import BaseEstimator, FrameTransformerMixin
from ._rule import Rule


class RuleEstimator(FrameTransformerMixin, BaseEstimator):
    def __init__(self, rule: dict):
        self.rule = rule

    def fit(self, X, y=None):
        self._ensure_dataframe(X)
        self._rule = Rule(self.rule)
        self._validate(X)
        self.n_features_in_= X.shape[1]
        return self
    
    def transform(self, X):
        self._validate(X)
        columns = self._rule.expr['output']['feature']
        if_part = self._rule.predict(X, 'if')
        then_part = self._rule.predict(X,'then')
        else_part = self._rule.predict(X,'else')
        total =[]
        if else_part[0] != else_part[0] and then_part[0] != then_part[0]: #如果then、else都为空的情况返回if的值
            return DataFrame(data=if_part, columns=[columns])
        elif else_part[0] != else_part[0] or then_part[0] != then_part[0]: #如果 then、else只有一部分将报错
            raise ValueError("then或者else有一个不存在")
        else:
            if type(then_part[0]) != type(else_part[0]):
                raise TypeError("then和else的类型必须一致")
            for i, j, k in zip(if_part, then_part, else_part):
                if i:
                    total.append(j)
                else:
                    total.append(k)
            return DataFrame(data=total, columns=[columns])
    
    #校验rule，转换成Rule实例
    def _validate(self, X):
        if isinstance(X, DataFrame):
            data = X.values
        elif isinstance(X,np.ndarray):
            data = X
        else:
            raise ValueError('Data type error!')
        return data
