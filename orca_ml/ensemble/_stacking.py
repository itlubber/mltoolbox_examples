from sklearn.ensemble import StackingClassifier, StackingRegressor
from ..base import ModelParameterProxy


class StackingClassifierParameterProxy(ModelParameterProxy):
    def _init_(self, estimators, final_estimator=None, cv=None, stack_method='auto', n_jobs=None, passthrough=False, verbose=0):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose

    def _make_estimator(self):
        estimator = StackingClassifier(estimators=self.estimators, final_estimator=self.final_estimator, cv=self.cv,
        stack_method=self.stack_method, n_jobs=self.n_jobs, passthrough=self.passthrough, verbose=self.verbose)
        self.estimator = estimator


class StackingRegressorParameterProxy(ModelParameterProxy):
    def init_(self, estimators, final_estimator=None, cv=None, n_jobs=None,
    passthrough=False, verbose=0):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose

    def _make_estimator(self):
        estimator = StackingRegressor(estimators=self.estimators, final_estimator=self.final_estimator, cv=self.cv,
                                      n_jobs=self.n_jobs, passthrough=self.passthrough, verbose=self.verbose)
        self.estimator = estimator
