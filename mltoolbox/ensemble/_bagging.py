from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from ..base import ModelParameterProxy


class BaggingClassifierParameterProxy(ModelParameterProxy):
    def __init__(self, base_estimator=None,
                n_estimators=10, *,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
                oob_score=False,
                warm_start=False,
                n_jobs=None,
                random_state=None,
                verbose=0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _make_estimator(self):
        estimator = BaggingClassifier(base_estimator=self.base_estimator,
                                        n_estimators=self.n_estimators,
                                        max_samples=self.max_samples,
                                        bootstrap=self.bootstrap,
                                        bootstrap_features=self.bootstrap_features,
                                        oob_score=self.oob_score,
                                        warm_start=self.warm_start,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state,
                                        verbose=self.verbose)
        self.estimator = estimator


class BaggingRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, base_estimator=None,
                n_estimators=10, *,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
                oob_score=False,
                warm_start=False,
                n_jobs=None,
                random_state=None,
                verbose=0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
    
    def _make_estimator(self):
        estimator = BaggingRegressor(base_estimator=self.base_estimator,
                                        n_estimators=self.n_estimators,
                                        max_samples=self.max_samples,
                                        max_features=self.max_features,
                                        bootstrap=self.bootstrap,
                                        bootstrap_features=self.bootstrap_features,
                                        oob_score=self.oob_score,
                                        warm_start=self.warm_start,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state,
                                        verbose=self.verbose)
        self.estimator = estimator
