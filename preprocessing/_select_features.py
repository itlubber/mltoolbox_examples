import numpy as np
from sklearn.utils.validation import check_is_fitted
from ..base import BaseEstimator, FrameTransformerMixin


class SelectFeatures(BaseEstimator, FrameTransformerMixin):
    def __init__(self, features=None):
        self.features = None
        if features:
            if isinstance(features, (list, np.ndarray)):
                self.features = features
            else:
                self.features = [features]

    def fit(self, X, y=None, **fit_params):
        self._check_n_features(X, reset=True)
        X = self._ensure_dataframe(X)
        self._check_params(X)
        return self

    def _check_params(self, X):
        if self.features is None:
            raise ValueError("features should not be None.")
        if not isinstance(self.features, list):
            raise TypeError("features should be a list.")
        if len(self.features) == 0:
            raise ValueError("features should not be empty.")
        # if not all(isinstance(f, str) for f in self.features):
        #     raise TypeError("features should be a list of str.")
        columns = X.columns.tolist()
        self.features_idx_ = [columns.index(f) for f in self.features]
    
    def transform(self, x):
        check_is_fitted(self, "features_idx_")
        self._check_n_features(X, reset=False)
        X = self._ensure_dataframe(X)
        return X[self.features]

    def fit_transform(self, X, y=None, **fit_params):
        self._check_n_features(X, reset=True)
        X = self._ensure_dataframe(X)
        self._check_params(X)
        return X[self.features]

    def _more_tags(self):
        return {
            "allow_nan": True,
        }
