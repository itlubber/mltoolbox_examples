from sklearn.ensemble._voting import VotingClassifier, VotingRegressor, _BaseVoting, _BaseHeterogeneousEnsemble
from ..base import ModelParameterProxy


class VotingClassifierParameterProxy(ModelParameterProxy):
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=None, flatten_transform=True):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform

    def _make_estimator(self):
        estimator = VotingClassifier(estimators=self.estimators, voting=self.voting, weights=self.weights, 
                                     n_jobs=self.n_jobs, flatten_transform=self.flatten_transform)
        self.estimator = estimator


class VotingRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, estimators, weights=None, n_jobs=None):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs

    def _make_estimator(self):
        estimator = VotingRegressor(estimators=self.estimators, weights=self.weights,
        n_jobs=self.n_jobs)
        self.estimator = estimator
