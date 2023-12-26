import numpy as np
from pandas import DataFrame, concat
from scipy import sparse
from sklearn.impute._base import MissingIndicator as SK_MissingIndicator
from sklearn.impute._base import SimpleImputer as SK_SimpleImputer
from sklearn.impute._base import _BaseImputer as _SK_BaseImputer
from sklearn.utils.validation import check_is_fitted


class _BaseImputer(_SK_BaseImputer):
    def _concatenate_indicator(self, X_imputed, X_indicator):
        """Concatenate indicator mask with the imputed data

        Parameters
        -----------
        self : instance of _BaseImputer
        X_imputed : array-like
        X_indicator : array-like

        Returns
        -----------
        Xt: array-like
        """
        if not self.add_indicator:
            return X_imputed
    
        if X_indicator is None:
            raise ValueError("Data from the missing indicator are not provided. Call _fit_indicator and _transform_indicator in the imputer implementation.")

        hstack = sparse.hstack if sparse.issparse(X_imputed) else np.hstack
        if isinstance(X_imputed, DataFrame):
            return concat([X_imputed, X_indicator], axis=1)
        else:
            return hstack( (X_imputed, X_indicator))


class MissingIndicator(SK_MissingIndicator):
    def transform(self, X):
        """Generate missing values indicator for X.

        Parameters
        -----------
        self : instance of MissingIndicator
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        
        Returns
        -----------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features) or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of  xt will be boolean.
        
        Examples
        -----------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from mltoolbox.impute import MissingIndicator
        >>> X = pd.DataFrame([[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]], co1umns=["f1", "f2", "f3"])
        >>> mi = MissingIndicator(features='missing-only').fit(x)
        >>> mi.transform(X)
        """
        if not isinstance(X, DataFrame):
            return super().transform(X)
        
        # if x is Dataframe
        columns = np.asarray(["%s_is_missing" % f for f in X.columns])
        index = X.index
        check_is_fitted(self)
        X = self._validate_input(X, in_fit=False)

        if X.shape[1] != self._n_features:
            raise ValueError ("X has a different number of features than during fitting.")
        
        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if self.error_on_new and features_diff_fit_trans.size > 0:
                raise ValueError("The features {} have missing values in transform but have no missing values in fit.".format(features_diff_fit_trans))
            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]
                columns = columns[self.features_]
        return DataFrame(data=imputer_mask, columns=columns, index=index)

    def fit_transform(self, X, y=None):
        """Generate missing values indicator for x.

        Parameters
        -----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -----------
        self : instance of MissingIndicator
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features) or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of Xt will be boolean.

        Examples
        -----------
        >>> import pandas as pd
        >>> from mltoolbox.impute import MissingIndicator
        >>> X = pd.DataFrame ( [[np .nan, 2, 3], [4, пр.nan, 6], [10, np.nan, 9]], columns=["f1", "f2", "f3"])
        >>> mi = MissingIndicator(features='missing-only')
        >>> mi.fit_transform(X)
        """
        if not isinstance(X, DataFrame):
            return super().fit_transform(self, X, y)
        index = X.index
        columns = np.asarray(["%s_is_missing" % f for f in X.columns])
        imputer_mask = self._fit(X, y)
        if self.features_.size < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]
            columns = columns[self.features_]
        return DataFrame(data=imputer_mask, columns=columns, index=index)


class SimpleImputer(SK_SimpleImputer):
    def _concatenate_indicator(self, X_imputed, X_indicator):
        """Concatenate indicator mask with the imputed data

        Parameters
        -----------
        self : instance of_BaseImputer
        X_imputed : array-like
        X_indicator : array-like

        Returns
        -----------
        Xt : array-like
        """
        if not self.add_indicator:
            return X_imputed
        
        if X_indicator is None:
            raise ValueError("Data from the missing indicator are not provided. Call _fit_indicator and _transform_indicator in the imputer implementation.")
        hstack = sparse.hstack if sparse.issparse(X_imputed) else np.hstack
        if isinstance(X_imputed, DataFrame):
            return concat([X_imputed, X_indicator], axis=1)
        else:
            return hstack((X_imputed, X_indicator))

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        -----------
        self : instance of SimpleImputer
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Examples
        -----------
        >>> import pandas as pd
        >>> from mltoolbox.impute import SimpleImputer
        >>> X = pd.DataFrame([[np.nan, 2, 3], [4, np.nan, 6],[1o,np.nan,9]],co1umns=["f1","f2","f3"])
        >>> imp = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True).fit(x)
        >>> imp.transform(X)
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


def is_imputer(obj):
    return issubclass(obj, _SK_BaseImputer) or isinstance(obj, _SK_BaseImputer)


def isa_imputer(est):
    return isinstance(est, _SK_BaseImputer)
