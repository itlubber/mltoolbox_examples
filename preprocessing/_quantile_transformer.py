import warnings
from pandas import DataFrame
from sklearn.preprocessing import QuantileTransformer as SK_QuantileTransformer


class QuantileTransformer(SK_QuantileTransformer):
    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        -----------
        self : sklearn.preprocessing.QuantileTransformer
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse matrix is provided, it will be converted into a sparse
            csc_matrix. Additionally, the sparse matrix needs to be nonnegative if ignore_implicit_zeros is False.

        Returns
        -----------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        if not isinstance(X, DataFrame):
            return super().transform(X)
        data = super().transform(X)
        columns = X.columns
        index = X.index
        return DataFrame(data=data, columns=columns, index=index)
    
    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        -----------
        self : sklearn.preprocessing.QuantileTransformer
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse matrix is provided, it ill be converted into a sparse
            csc_matrix. Additionally, the sparse matrix needs to be nonnegative if ignore_implicit_zeros is False
        
        Returns
        -----------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        if not isinstance(X, DataFrame):
            return super().inverse_transform(X)
        data = super().inverse_transform(X)
        columns = X.columns
        index = X.index
        return DataFrame(data=data, columns=columns, index=index)


def quantile_transform(X, *, axis=0, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, subsample=int(1e5), random_state=None, copy="warn"):
    """
    Examples
    -----------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from orca_ml.preprocessing import quantile_transform
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> X = pd.DataFrame(X)
    >>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
    """
    if copy == "warn":
        copy = False
    n = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, subsample=subsample,
                            ignore_implicit_zeros=ignore_implicit_zeros, random_state=random_state, copy=copy)
    if axis == 0:
        return n.fit_transform(X)
    elif axis == 1:
        return n.fit_transform(X.T).T
    else:
        raise ValueError("axis should be either equal to o or 1. Got axis={}".format(axis))
