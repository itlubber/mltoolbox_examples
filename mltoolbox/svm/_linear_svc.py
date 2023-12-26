from sklearn.svm._classes import LinearSVC, LinearSVR

from ..base import ModelParameterProxy


class LinearSVCParameterProxy(ModelParameterProxy):
    def __init__(self, penalty='12', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr',
                fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                max_iter=1000):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

    def _make_estimator(self):
        estimator = LinearSVC(
                                penalty=self.penalty,
                                loss=self.loss,
                                tol=self.tol,
                                C=self.C,
                                multi_class=self.multi_class,
                                fit_intercept=self.fit_intercept,
                                verbose=self.verbose,
                                intercept_scaling=self.intercept_scaling,
                                class_weight=self.class_weight,
                                random_state=self.random_state,
                                max_iter=self.max_iter
                                )
        self.estimator = estimator


class LinearSVRParameterProxy(ModelParameterProxy):
    def __init__(self, epsilon=0.0, tol=1e-4, C=1.0, loss='epsilon_insensitive', fit_intercept=True,
                 intercept_scaling=1., dual=True, verbose=0, random_state=None, max_iter=1000):
        self.epsilon = epsilon
        self.tol = tol
        self.C = C
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.dual = dual
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

    def _make_estimator(self):
        estimator = LinearSVR(
                                epsilon=self.epsilon,
                                tol=self.tol,
                                C=self.C,
                                loss=self.loss,
                                fit_intercept=self.fit_intercept,
                                intercept_scaling=self.intercept_scaling,
                                dual=self.dual,
                                verbose=self.verbose,
                                max_iter=self.max_iter,
                                random_state=self.random_state,
                            )
        self.estimator = estimator
