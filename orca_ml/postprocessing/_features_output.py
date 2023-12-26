from sklearn.utils.validation import check_is_fitted
from ..base import BaseEstimator, FrameTransformerMixin


class FeaturesOutPut(BaseEstimator, FrameTransformerMixin):
    def __init__(self, positions=None):
        self.features_idx_ = positions

    def fit(self, X, y=None, **fit_params):
        self._check_n_features(X, reset=True)
        X = self._ensure_dataframe(X)
        # columns = X.columns.tolist()
        # self.features_idx_ = [columns.index(f) for f in self.positions]
        # self._check_params(X)
        return self

    def _check_params(self, X):
        if self.features_idx_:
            if not isinstance(self.features_idx_, list):
                raise TypeError("`positions` should be a list or None")
            if len(self.features_idx_) == 0:
                raise ValueError("`positions` should not be empty.")
            if not all(isinstance(f, int) for f in self.features_idx_):
                raise TypeError("`positions` should be a list of int.")
        else:
            self.features_idx_ = [x for x in range(X.shape[1])]

    def transform(self, X):
        check_is_fitted(self, "features_idx_")
        self._check_n_features(X, reset=False)
        X = self._ensure_dataframe(X)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self._check_n_features(X, reset=True)
        X = self._ensure_dataframe(X)
        self._check_params(X)
        return X

    def features_output(self, X, y=None, **fit_params):
        if self.features_idx_ is None:
            return X
        else:
            return X.iloc[:, self.features_idx_]
    
    def _more_tags(self):
        return {
            "allow_nan": True,
        }
