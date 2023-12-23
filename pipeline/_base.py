import numpy as np
from joblib import Parallel, delayed
from pandas import DataFrame, concat
from scipy import sparse
from sklearn.pipeline import FeatureUnion as SK_FeatureUnion
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import _transform_one, _fit_transform_one, _name_estimators


class FeatureUnion(SK_FeatureUnion):
    def transform(self, X):
        """Transform X separately by each transformer, concatenate results

        Parameters
        -----------
        self : sklearn.pipeline.FeatureUnion
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        
        Returns
        -----------
        Xs : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the sum of n_components (output dimension) over transformers.
        
        Examples
        -----------
        >>> import orca_ml
        >>> import pandas as pd
        >>> from orca_ml.pipeline import FeatureUnion
        >>> from orca_ml.decomposition import PCA, TruncatedsVD
        >>> X = pd.DataFrame([[0., 1., 3], [2., 2., 5]])
        >>> union = FeatureUnion([("pca", PCA(n_components=1)), ("svd", TruncatedsVD(n components=2))]).fit(x)
        >>> union.transform(X)
        pca_component1 truncated_svd_componentil truncated_svd_component2
         1.5           3.039550                  0.872432
        -1.5           5.725864                 -0.463127
        """
        Xs = Parallel(n_jobs=self.n_jobs)(delayed(_transform_one)(trans, X, None, weight) for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            data = np.zeros((X.shape[0], 0))
            if isinstance(X, DataFrame):
                return DataFrame(data=data, index=X.index, columns=X.columns)
            else:
                return data
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        elif all(isinstance(f, DataFrame) for f in Xs):
            Xs = concat(Xs, axis=1)
        else:
            Xs = np.hstack(Xs)
        return Xs
    
    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        -----------
        self: sklearn.pipeline.FeatureUnion
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        fit_params : dict
        
        Returns
        -----------
        Xs : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the sum of n_components (output dimension) over transformers.
        
        Examples
        -----------
        >>> import orca_m1
        >>> import pandas as pd
        >>> from orca_ml.pipeline import FeatureUnion
        >>> from orca_ml.decomposition import PCA, TruncatedsvD
        >>> X = pd.DataFrame([[0., 1., 3], [2., 2., 5]])
        >>> union = FeatureUnion([("pca", PCA(n_components=1)), ("svd", TruncatedsVD(n_components=2))])
        >>> union.fit_transform(union, x)
        pca_componenti truncated_svd_componentl truncated_svd_component2
         1.5           3.039550                 0.872432
        -1.5           5.725864                -0.463127
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            data = np.zeros((X. shape[0], 0))
            if isinstance(X, DataFrame):
                return DataFrame(data=data, index=X.index, columns=X.columns)
            else:
                return data
            
        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        elif all(isinstance(f, DataFrame) for f in Xs):
            Xs = concat(Xs, axis=1)
        else:
            Xs = np.hstack(Xs)
        return Xs


def make_union(*transformers, n_jobs=None, verbose=False):
    """Construct a FeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require, and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.

    Parameters
    -----------
    *transformers : list of estimators

    n_jobs : int, default=None
        Number of jobs to run in parallel
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more details.
    
        .. versionchanged:: v0.20
            `n_jobs` default changed from 1 to None
    
    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be printed as it is completed.
    
    Returns
    -----------
    f: FeatureUnion

    See Also
    -----------
    FeatureUnion : Class for concatenating the results of multiple transformer objects.
        
    Examples
    -----------
    >>> from orca_ml.decomposition import PCA, TruncatedsVD
    >>> from orca_ml.pipeline import make_union
    >>> make_union(PCA(), TruncatedsVD())
    FeatureUnion(transformer_list=[('pca', PCA()), ('truncatedsvd', TruncatedsVD())])
    """
    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
