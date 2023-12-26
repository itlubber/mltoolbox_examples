import numpy as np
from sklearn.linear_model._base import LinearRegression
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator, RegressorMixin


class OLSLinearRegression(BaseEstimator, RegressorMixin):
    """
    Examples
    ----------
    >>> import numpy as np
    >>> from orca_ml.linear_model._base import OLSLinearRegression
    >>> X = np.array([ 1.0, 2.1, 3.6, 4.2, 6])[:, np.newaxis]
    >>> y = np.array([ 1.0, 2.0, 3.0, 4.0, 5.0])
    >>> ne_lr = OLSLinearRegression(method='qr')
    >>> ne_lr.fit(X, y)
    OLSLinearRegression(method='qr')
    """
    def __init__(self, method="direct", eta=0.01, epochs=50, n_batches=None, random_state=None, verbose=0):
        self.method = method
        self.eta = eta
        self.epochs = epochs
        self.n_batches = n_batches
        self.random_state = random_state
        self.verbose = verbose

    def _check_X(self, X):
        X = check_array(X, dtype="numeric", force_all_finite=True)
        return X
    
    def fit(self, X, y, init_params=True):
        if self.method != 'sgd' and self.n_batches is not None:
            raise ValueError("`n_batches` should be set to None if `method` != 'sgd'. Got method={}".format(self.method))
        support_methods = ('sgd', 'direct', 'svd', 'qr')
        if self.method not in support_methods:
            raise ValueError("`method` must be in {}. Got {}.".format(support_methods, self.method))
        return self._fit(X, y, init_params=init_params)
    
    def _fit(self, X, y, init_params=True):
        X = self._check_X(X)
        random_state = check_random_state(self.random_state)
        if init_params:
            self.coef_, self.intercept_ = init_weight_bias((X.shape[1],), bias_shape=(1,), random_state=random_state)
            self.losses_ = []

        if self.method == 'direct':
            self.coef_, self.intercept_ = solve_normal_equation(X, y)
        elif self.method == 'sgd':
            losses = []
            for i in range(self.epochs):
                for idx in _yield_batch_indices(random_state, n_batches=self.n_batches, arr=y, shuffle=True):
                    y_pred = self._forward(X[idx])
                    errors = (y[idx] - y_pred)
                    self.coef_ += self.eta * X[idx].T.dot(errors)
                    self.intercept_ += self.eta * errors.sum()

                loss = sum_square_error(y, self._forward(X))
                if self.verbose:
                    print("Iteration: %d/%d | Loss %.2f".format(i + 1, self.epochs, loss))
                losses.append(loss)
            self.losses_ = losses
        elif self.method == 'qr':
            Xb = np.hstack((np.ones((X.shape[0], 1)), X))
            Q, R = np.linalg.qr(Xb)
            beta = np.dot(np.linalg.inv(R), np.dot(Q.T, y))
            self.coef_ = beta[1:]
            self.intercept_ = np.array([beta[0]])
        elif self.method == 'svd':
            Xb = np.hstack((np.ones((X.shape[0], 1)), X))
            beta = np.dot(np.linalg.pinv(Xb), y)
            self.coef_ = beta[1:]
            self.intercept_ = np.array([beta[0]])

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = self._check_X(X)
        return self._forward(X)

    def _forward(self, X):
        return np.dot(X, self.coef_) + self.intercept_


def init_weight_bias(weights_shape, bias_shape=(1,), random_state=None, dtype='float64', scale=0.01, bias_const=0.0):
    w = random_state.normal(loc=0.0, scale=scale, size=weights_shape)
    b = np.zeros(shape=bias_shape)
    if bias_const != 0.0:
        b += bias_const
    return w.astype(dtype), b.astype(dtype)


def solve_normal_equation(X, y):
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    params = np.dot(z, np.dot(Xb.T, y))
    w, b = np.ravel(params[1:]), np.array([params[0]])
    return w, b


def sum_square_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / 2.0


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
