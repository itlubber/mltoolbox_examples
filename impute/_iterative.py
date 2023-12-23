import numpy as np
from pandas import DataFrame
from sklearn.impute._iterative import IterativeImputer as SK_IterativeImputer


class IterativeImputer(SK_IterativeImputer):
    def transform(self, X):
        """Imputes all missing values in X.
        
        Note that this is stochastic, and that if random_state is not fixed, repeated calls, or permuted input, will yield different results.
        
        Parameters
        -----------
        self : sklearn.impute._iterative. IterativeImputer
        X : array-like of shape (n_samples, n_features)
            The input data to complete.
        
        Returns
        -----------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        if not isinstance(X, DataFrame):
            return super().transform(X)
        data = super().transform(X)
        index = X.index
        columns = X.columns
        if self.add_indicator:
            ind_cols = np.asarray(["%s_is_missing" % f for f in columns])
            indicator_ = self.indicator_
            if indicator_.features == "missing-only":
                if indicator_.features_.size < indicator_._n_features:
                    ind_cols = ind_cols[indicator_.features_]
            return DataFrame(data=data, columns=[*columns, *ind_cols], index=index)
        return DataFrame(data=data, columns=columns, index=index)

    def fit_transform(self, X, y=None):
        """Fits the imputer on X and return the transformed X.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and "n_features" is the number of features.
        y : ignored.

        Returns
        -----------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        if not isinstance(X, DataFrame):
            return super().fit_transform(self, X, y)
        data = super().fit_transform(self, X, y)
        index = X.index
        columns = X.columns
        if self.add_indicator:
            ind_cols= np.asarray(["%s_is_missing" %f for f in columns])
            indicator_ = self.indicator_
            if indicator_.features == "missing-only":
                if indicator_.features_.size < indicator_._n_features:
                    ind_cols = ind_cols[indicator_.features_]
            return DataFrame(data=data, columns=[*columns, *ind_cols], index=index)
        return DataFrame(data=data, columns=columns, index=index)
