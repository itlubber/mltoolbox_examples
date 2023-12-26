try:
    from sklearn.ensemble.iforest import IsolationForest
except ImportError:
    from sklearn.ensemble._iforest import IsolationForest

from ..base import ModelParameterProxy


class IsolationForestParameterProxy(ModelParameterProxy):
    def _init__(self, n_estimators=100,
                max_samples="auto",
                contamination="auto",
                max_features=1.,
                bootstrap=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

    def _make_estimator(self):
        estimator = IsolationForest(n_estimators=self.n_estimators,
                                    max_samples=self.max_samples,
                                    contamination=self.contamination,
                                    max_features=self.max_features,
                                    bootstrap=self.bootstrap,
                                    n_jobs=self.n_jobs,
                                    random_state=self.random_state,
                                    verbose=self.verbose,
                                    warm_start=self.warm_start)
        self.estimator = estimator
