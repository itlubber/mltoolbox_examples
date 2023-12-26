from sklearn.preprocessing import FunctionTransformer


def transform_function_transformer(self, X):
    """Transform X using the forward function.

    Parameters
    self : instance of FunctionTransformer
    X : array-like, shape (n_samples, n_features)
        Input array.
    
    Returns
    X_out : array-like, shape (n_samples, n_features)
    Transformed input.

    Examples
    >>> import numpy as np
    >>> import pandas as pd
    >>> from orca_ml.preprocessing import FunctionTransformer
    >>> X = pd.DataFrame([[0, 1], [2, 3]], columns-["f1", "f2"])
    >>> transformer = FunctionTransformer(func=np.exp).fit(x)
    >>> transform_function_transformer(transformer, x)
    """
    return self._transform(X, func=self.func, kw_args=self.kw_args)


def inverse_transform_function_transformer(self, X):
    """Transform X using the inverse function.

    Parameters
    self : instance of FunctionTransformer
    X : array-like, shape (n_samples, n_features)
        Input array.

    Returns
    X_out : array-like, shape (n_samples, n_features)
        Transformed input.

    Examples
    >>> import numpy as np
    >>> import pandas as pd
    >>> from orca_ml.preprocessing import FunctionTransformer
    >>> x = pd.DataFrame([[0, 1], [2, 3]l, columns=["f1", "f2"])
    >>> transformer = FunctionTransformer(func=np.exp, inverse_func=np.log).fit(x)
    >>> Xt = transform_function_transformer(transformer, x)
    >>> inverse_transform_function_transformer(transformer, xt)
    """
    return self._transform(X, func=self.inverse_func, kw_args=self.inv_kw_args)


FunctionTransformer.transform = transform_function_transformer
FunctionTransformer.inverse_transform = inverse_transform_function_transformer
