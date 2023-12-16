import numpy as np
from sklearn.utils import _deprecate_positional_args, check_array
from ._base import BaseDiscretizer


class CustomizedDiscretizer(BaseDiscretizer):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=None, n_jobs=None, bin_edges=None):
        super(CustomizedDiscretizer, self).__init__(n_bins=n_bins, n_jobs=n_jobs)
        self.bin_edges = bin_edges

    def fit(self, X, y=None, **fit_params):
        self._validate_params(X)
        return self
    
    def _validate_params(self, X):
        X = check_array(X, dtype="numeric")
        _, n_features = X.shape
        if self.bin_edges is None:
            raise ValueError("bin_edges should not be empty.")
        if len(self.bin_edges) != n_features:
            raise ValueError("number of bin_edges does not match number of features.")
        self.bin_edges_ = list(self.bin_edges)
        if self.n_bins is None:
            self.n_bins_ = np.asarray([len(bin_edge) - 1 for bin_edge in self.bin_edges_], dtype=int)
        else:
            n_bins_ = np.asarray([len(bin_edge) - 1 for bin_edge in self.bin_edges_], dtype=int)
            if not np.array_equal(self.n_bins, n_bins_):
                raise ValueError("n_bins should be consistent with bin_edges.")
            self.n_bins_ = n_bins_
