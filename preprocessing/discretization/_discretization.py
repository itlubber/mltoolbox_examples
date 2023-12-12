from pandas import DataFrame
from sklearn.preprocessing import KBinsDiscretizer


_all_= ["KBinsDiscretizer"]

_transform_kbins_discretizer = KBinsDiscretizer.transform


def transform_kbins_discretizer(self, X):
    """Discretize the data.

    Parameters
    ----------
    self : sklearn.preprocessing.KBinsDiscretizer
    X : numeric array-like, shape (n_samples, n_features)
        Data to be discretized.
    
    Returns
    ----------
    Xt : numeric array-like or sparse matrix
        Data in the binned space.
    """
    data = _transform_kbins_discretizer(self, X)
    if self.encode == 'onehot':
        data = data.toarray()
    if isinstance (X, DataFrame):
        columns = X.columns
        index = X.index
        return DataFrame(data=data, columns=columns, index=index).astype(dtype='category')
    return data


_inverse_transform_kbins_discretizer = KBinsDiscretizer.inverse_transform


def inverse_transform_kbins_discretizer(self, Xt):
    """Transform discretized data back to original feature space.

    Note that this function does not regenerate the original data due to discretization rounding.

    Parameters
    ----------
    self : sklearn.preprocessing.KBinsDiscretizer
    Xt : numeric array-like, shape (n_sample, n_features)
        Transformed data in the binned space.

    Returns
    ----------
    Xinv : numeric array-like
        Data in the original feature space.
    """
    data = _inverse_transform_kbins_discretizer(self, Xt)
    if isinstance(Xt, DataFrame):
        columns = Xt.columns
        index = Xt.index
        return DataFrame (data=data, columns=columns, index=index)
    return data


KBinsDiscretizer.transform = transform_kbins_discretizer
KBinsDiscretizer.inverse_transform = inverse_transform_kbins_discretizer
