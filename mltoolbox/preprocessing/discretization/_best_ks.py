import numpy as np
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.extmath import stable_cumsum
from ._base import BaseDiscretizer
from ...base import MetaEstimatorMixin


def _recurse(pre_bin_edges, n_events, n_nonevents, n_bins, bin_edges=None):
    if len(pre_bin_edges) - 1 <= 1 or len(bin_edges) - 1 >= n_bins:
        return None
    # assume Len(pre_bin_edges) = n_pre_bins + 1
    # Len(tpr) = Len(fpr) = n_pre_bins + 1
    tps = np.r_[0, stable_cumsum(n_events)]
    fps = np.r_[0, stable_cumsum(n_nonevents)]
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    ks_values = np.abs(tpr - fpr)
    idx = np.argmax(ks_values)
    # Len(pre_bin_edges) = Len(ks_values)
    best_split = pre_bin_edges[idx]
    bin_edges.append(best_split)
    
    pre_bin_edges_left = pre_bin_edges[:idx + 1]
    pre_bin_edges_right = pre_bin_edges[idx:]
    n_events_left, n_nonevents_left = n_events[:idx], n_nonevents[:idx]
    n_events_right, n_nonevents_right = n_events[idx:], n_nonevents[idx:]
    _recurse(pre_bin_edges_left, n_events_left, n_nonevents_left, n_bins, bin_edges=bin_edges)
    _recurse(pre_bin_edges_right, n_events_right, n_nonevents_right, n_bins, bin_edges=bin_edges)


class BestKSDiscretizer(BaseDiscretizer, MetaEstimatorMixin):
    """This class implements the Best-KS binning method.

    methodology: 1. prebinning the ascending data into designated bins
                 2. calculate the KS value of each bin, find the best KS idx = i
                 3. split the data into two sub-parts: left bins[0:i], right bins[i:-1]
                 4. for the resulted left bins and right bins, repeat the step2 and step3 recursively
                 5. end when conditions satisfied

    Parameters
    -----------
    estimator : prebinning discretizer instance.
                estimator used to prebin the data
    n_bins : int, bin num
    n_jobs : int, default=None
            applicable when multi-threading

    Examples
    -----------
    >>> from mltoolbox.datasets import load_uci_credit
    >>> from mltoolbox.preprocessing import BestKSDiscretizer, UniformDiscretizer
    >>> X, y = load_uci_credit(return_X_y=True, as_frame=True)
    >>> ks_discretizer = BestKSDiscretizer(UniformDiscretizer(n_bins=20), n_bins=5).fit(x, y)
    >>> ks_discretizer.transform(X)
    """
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_bins=5, n_jobs=None):
        self.estimator = estimator
        super(BestKSDiscretizer, self).__init__(n_bins=n_bins, n_jobs=n_jobs)

    def _validate_params(self):
        if not isinstance(self.estimator, BaseDiscretizer):
            raise ValueError("estimator should be of type `BaseDiscretizer`.")
        
    def fit(self, X, y=None, **fit_params):
        """Fit the binner .

        Parameters
        -----------
        X: numeric array-like, shape (n_samples,)
            Data to be discretized.
        у : numeric array-like, shape (n_samples,)
            best-ks is supervised estimator, and thus 'y' must be introduced
        
        Returns
        -----------
        self
        """
        self._validate_params()
        self.estimator.fit(X, y=y, **fit_params)
        return super (BestKSDiscretizer, self).fit(X, y=y)
    
    @property
    def closed(self):
        return self.estimator.closed
    
    @closed.setter
    def closed(self, value):
        self.estimator.closed = value
    
    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        pre_bin_edges  = self.estimator.bin_edges_[i]
        n_pre_bins = len(pre_bin_edges) - 1
        n_bin = min(n_pre_bins, n_bin)

        right = self.closed == 'right'
        indices = np.digitize(x, pre_bin_edges[1:-1], right=right)

        n_nonevents = np.empty(n_pre_bins).astype(np.int64)
        n_events = np.empty(n_pre_bins).astype(np.int64)

        for i in range(n_pre_bins):
            mask = (indices == i)
            bin_size = np.count_nonzero(mask)
            n_event = np.count_nonzero((y == 1) & mask)
            n_events[i] = n_event
            n_nonevents[i] = bin_size - n_event

        bin_edges = [pre_bin_edges[0], pre_bin_edges[-1]]
        _recurse(pre_bin_edges, n_events, n_nonevents, n_bin, bin_edges=bin_edges)
        bin_edges = np.sort(bin_edges)
        return bin_edges, len(bin_edges) - 1
    
    def _more_tags(self):
        return {
            "X_types": _safe_tags(self.estimator, "X_types"),
            "allow_nan": _safe_tags(self.estimator, "allow_nan"),
            "requires_y": True,
        }
