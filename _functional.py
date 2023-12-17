from .base import ModelParameterProxy
from typing import Callable


class BaseFitPredictFunction(Callable):
    proxy_class = ModelParameterProxy

    def __init__(self, *args, **kwargs):
        proxy_instance = self.proxy_class(*args, **kwargs)
        proxy_instance._make_estimator()
        self.estimator = proxy_instance.estimator

    def __call__(self, X, y=None, **fit_params):
        self.estimator.fit(X, y=y, **fit_params)
        return self.estimator.predict(X)
