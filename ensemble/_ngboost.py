from ngboost import NGBoost
from ngboost import NGBClassifier, NGBRegressor, NGBSurvival
from ngboost.distns import Bernoulli, Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import LogScore

from ..base import ModelParameterProxy


def has_ngboost():
    try:
        import ngboost
        return True
    except ImportError:
        return False


def isa_ngboost_model(clf):
    if not has_ngboost():
        return False
    from ngboost import NGBoost # noqa: F401
    return isinstance(clf, NGBoost)


def is_ngboost_model(cls):
    if not has_ngboost():
        return False
    from ngboost import NGBoost # noqa: F401
    return issubclass(cls, NGBoost)


class NGBClassifierParameterProxy(ModelParameterProxy):
    def __init__(self, Dist=Bernoulli, Score=LogScore, Base=default_tree_learner, natural_gradient=True, n_estimators=500,
                 learning_rate=0.01, minibatch_frac=1.0, col_sample=1.0, verbose=True, verbose_eval=100, tol=1e-4, random_state=None):
        self.Dist = Dist
        self.Score = Score
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.tol = tol
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.verbose_eval = verbose_eval
        self.random_state = random_state

    def _make_estimator(self):
        return NGBClassifier(
            Dist=self.Dist,
            Score=self.Score,
            Base=self.Base,
            natural_gradient=self.natural_gradient,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            tol=self.tol,
            random_state=self.random_state,
        )


class NGBRegressorParameterProxy(ModelParameterProxy):
    def __init__(self, Dist=Normal, Score=LogScore, Base=default_tree_learner, natural_gradient=True, n_estimators=500,
                 learning_rate=0.01, minibatch_frac=1.0, col_sample=1.0, verbose=True, verbose_eval=100, tol=1e-4, random_state=None):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.tol = tol
        self.random_state = random_state

    def _make_estimator(self):
        return NGBRegressor(
            Score=self.Score,
            Dist=self.Dist,
            Base=self.Base,
            natural_gradient=self.natural_gradient,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            tol=self.tol,
            random_state=self.random_state,
        )
