import numpy as np
from pandas import DataFrame
from sklearn.impute import KNNImputer as SK_KNNImputer


class KNNImputer(SK_KNNImputer):
    def transform(self, X):
        """Impute all missing values in x.

        Parameters
        -----------
        self : sklearn.impute.KNNImputer
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -----------
        X : array-like of shape (n_samples, n_output_features)
            The imputed dataset. `n_output_features` is the number of features that is not always missing during fit.

        Examples
        -----------
        >>> import pandas as pd
        >>> from orca_ml.impute import KNNImputer
        >>> X = pd.DataFrame ([[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]], columns=["f1", "f2", "f3"])
        >>> imp = KNNImputer(missing_values=np.nan, add_indicator=True).fit(x)
        >>> transform_knn_imputer(imp, X)
        """
        if not isinstance(X, DataFrame):
            return super(). transform(X)
        data = super().transform(X)
        columns = X.columns
        index = X.index
        if self.add_indicator:
            ind_cols = np.array(["%s_is_missing" % f for f in columns])
            indicator_ = self.indicator_
            if indicator_.features == "missing-only":
                if indicator_.features_.size < indicator_._n_features:
                    ind_cols = ind_cols[indicator_.features_]
            return DataFrame(data=data, columns=[*columns, *ind_cols], index=index)
        return DataFrame(data=data, columns=columns, index=index)
