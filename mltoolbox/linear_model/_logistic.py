import numpy as np
from sklearn.linear_model._logistic import LogisticRegression, LogisticRegressionCV
from sklearn.utils.validation import check_random_state, check_array, check_is_fitted
from ._utils import _rescale_data
from ..base import BaseEstimator, ClassifierMixin, ModelParameterProxy


class RandomizedLogisticRegression(LogisticRegression):
    """Randomized version of scikit-learn LogisticRegression class.

    Randomized LASSO is a generalization of the LASSo. The LASSO penalises the absolute value of the coefficients with a penalty
    term proportional to `C`, but the randomized LASSo changes the penalty to a randomly chosen value value in range `[C, C/weakness]`

    Parameters
    ----------
    weakness : float, default = 0.5
        Weakness value for randomized LASSO. Must be in (0, 1].
    
    See Also
    ----------
    mltoolbox.linear_model.LogisticRegression : learns logistic regression models using the same algorithm.
    """
    def __init__(self, *, weakness=0.5, penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        self.weakness = weakness
        super(RandomizedLogisticRegression, self).__init__(
                penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                l1_ratio=l1_ratio,
            )

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples, n_features)
            The target values.
        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.

        Returns
        ----------
        self
        """
        if not isinstance(self.weakness, float) or not (0.0 < self.weakness <= 1.0):
            raise ValueError('weakness should be a float in (0, 1], got %s' % self.weakness)
        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=[np.float64, np.float32], order="C")
        n_features = X.shape[1]
        random_state = check_random_state(self.random_state)
        weakness = 1. - self.weakness
        weights = weakness * random_state.randint(0, 1 + 1, size=(n_features,))
        X = _rescale_data(X, weights)
        return super(RandomizedLogisticRegression, self).fit(X, y, sample_weight=sample_weight)


class LogisticRegressionParameterProxy(ModelParameterProxy):
    """
    Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the
    cross-entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag', 'saga' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers. **Note
    that regularization is applied by default**. It can handle both dense
    and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit
    floats for optimal performance; any other input format will be converted
    (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation, or no regularization. The 'liblinear' solver
    supports both L1 and L2 regularization, with a dual formulation only for
    the L2 penalty. The Elastic-Net regularization is only supported by the
    'saga' solver.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Specify the norm of the penalty:

        - `None`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

        .. deprecated:: 1.2
           The 'none' option was deprecated in version 1.2, and will be removed
           in 1.4. Use `None` instead.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data. See :term:`Glossary <random_state>` for details.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
            default='lbfgs'

        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:

            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' is limited to one-versus-rest schemes.
            - 'newton-cholesky' is a good choice for `n_samples` >> `n_features`,
              especially with one-hot encoded categorical features with rare
              categories. Note that it is limited to binary classification and the
              one-versus-rest reduction for multiclass classification. Be aware that
              the memory usage of this solver has a quadratic dependency on
              `n_features` because it explicitly computes the Hessian matrix.

        .. warning::
           The choice of the algorithm depends on the penalty chosen.
           Supported penalties by solver:

           - 'lbfgs'           -   ['l2', None]
           - 'liblinear'       -   ['l1', 'l2']
           - 'newton-cg'       -   ['l2', None]
           - 'newton-cholesky' -   ['l2', None]
           - 'sag'             -   ['l2', None]
           - 'saga'            -   ['elasticnet', 'l1', 'l2', None]

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on features
           with approximately the same scale. You can preprocess the data with
           a scaler from :mod:`sklearn.preprocessing`.

        .. seealso::
           Refer to the User Guide for more information regarding
           :class:`LogisticRegression` and more specifically the
           :ref:`Table <Logistic_regression>`
           summarizing solver/penalty supports.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.
        .. versionchanged:: 0.22
            The default solver changed from 'liblinear' to 'lbfgs' in 0.22.
        .. versionadded:: 1.2
           newton-cholesky solver.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.22
            Default changed from 'ovr' to 'auto' in 0.22.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See :term:`Glossary <n_jobs>` for more details.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression (when given
        the parameter ``loss="log_loss"``).
    LogisticRegressionCV : Logistic regression with built-in cross validation.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------

    L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
        http://users.iems.northwestern.edu/~nocedal/lbfgsb.html

    LIBLINEAR -- A Library for Large Linear Classification
        https://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
            :arxiv:`"SAGA: A Fast Incremental Gradient Method With Support
            for Non-Strongly Convex Composite Objectives" <1407.0202>`

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from mltoolbox.linear_model._logistic import LogisticRegressionParameterProxy
    >>> X, y = load_iris(return_X_y=True)
    >>> wrapper = LogisticRegressionParameterProxy(random_state=0)
    >>> wrapper._make_estimator()
    >>> clf = wrapper.estimator.fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(X, y)
    0.97...
    """
    def __init__(self, penalty='12', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None,
                 solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def _make_estimator(self):
        estimator = LogisticRegression(
            penalty=self.penalty, dual=self.dual, tol=self.tol, C=self.C, fit_intercept=self.fit_intercept, intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight, random_state=self.random_state, solver=self.solver, max_iter=self.max_iter, multi_class=self.multi_class, 
            verbose=self.verbose, warm_start=self.warm_start, n_jobs=self.n_jobs, l1_ratio=self.l1_ratio)
        self.estimator = estimator


class BinaryLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Examples
    --------
    >>> from mltoolbox.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> x = XΓ:, [0, 31] # sepal length and petal width
    >>> X = X[0:100] # class 0 and class 1
    >>> y = y[0:100] # class 0 and class 1
    >>> # standardize
    >>> X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    >>> X[:,1] = (X[:,1] -X[:,1].mean()) X[:,1].std()
    >>> from mltoolbox.linear_model._logistic import BinaryLogisticRegression
    >>> ada = BinaryLogisticRegression(epochs=30, eta=0.01, n_batches=1, random_state=1)
    >>> ada.fit(X, y)
    BinaryLogisticRegression(epochs=30, random_state=1)
    """
    def _init_(self, eta=0.01, epochs=50, l2_lambda=0.0, n_batches=1, random_state=None, verbose=0):
        self.eta = eta
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.n_batches = n_batches
        self.random_state = random_state
        self.verbose = verbose

    def _check_X(self, X):
        X = check_array(X, dtype="numeric", force_all_finite=True)
        return X
    
    def _check_y(self, y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        if len(le.classes_) != 2:
            raise ValueError("Only support binary label for bad rate encoding!")
        self.classes_ = le.classes_
        return y

    def fit(self, X, y, **fit_params):
        X, y = self._check_X(X), self._check_y(y)
        return self._fit(X, y, **fit_params)
    
    def _fit(self, X, y, **fit_params):
        random_state = check_random_state(self.random_state)
        if self.n_batches is None:
            raise ValueError("n_batches should not be None!")
        losses = []
        self.coef_, self.intercept_= init_weight_bias(weights_shape=(X.shape[1],), bias_shape=(1,), random_state=random_state)
        for i in range(self.epochs):
            for idx in _yield_batch_indices(random_state, n_batches=self.n_batches, arr=y, shuffle=True):
                y_pred = self._activation(X[idx])
                errors = (y[idx] - y_pred)
                self.coef_ += self.eta * (X[idx].T.dot(errors) - self.l2_lambda * self.coef_)
                self.intercept_ += self.eta * errors.sum()

            loss = self._logit_loss(y, self._activation(X))
            losses.append(loss)
            if self.verbose:
                print("Iteration: %d/%d | Loss %.2f".format(i + 1, self.epochs, loss))
        return self

    def _forward(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def _activation(self, X):
        return sigmoid(self._forward(X))
    
    def _logit_loss(self, y_true, y_pred):
        logit = -y_true.dot(np.log(y_pred)) - (1 - y_true).dot(np.log(1 - y_pred))
        if self.l2_lambda:
            logit += self.l2_lambda / 2.0 * np.sum(self.coef_ ** 2)
        return logit

    def predict_proba(self, X):
        return self._activation(X)

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept "])
        X = self._check_X(X)
        # Here we add a threshold function to convert the continuous outcome to a categorical class
        return np.where(self._forward(X) < 0.0, 0, 1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_weight_bias(weights_shape, bias_shape=(1,), random_state=None, dtype='float64', scale=0.01, bias_const=0.0):
    w = random_state.normal(loc=0.0, scale=scale, size=weights_shape)
    b = np.zeros(shape=bias_shape)
    if bias_const != 0.0:
        b += bias_const
    return w.astype(dtype), b.astype(dtype)


def _yield_batch_indices(gen, n_batches, arr, shuffle=True):
    n_samples = arr.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        indices = gen.permutation(indices)
    if n_batches > 1:
        remainder = n_samples % n_batches
        if remainder:
            minis = np.array_split(indices[:-remainder], n_batches)
            minis[-1] = np.concatenate((minis[-1], indices[-remainder:]), axis=0)
        else:
            minis = np.array_split(indices, n_batches)
    else:
        minis = (indices,)
    for idx_batch in minis:
        yield idx_batch
