import numpy as np
from scipy.stats import chi2_contingency
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import _deprecate_positional_args
from ._base import BaseDiscretizer
from ...base import MetaEstimatorMixin


class ChiMergeDiscretizer(BaseDiscretizer, MetaEstimatorMixin):
    """This class implements the chi-merge binning method

    methodology: 1. prebinning the ascending data into designated bins
                 2. Calculate the chi2 score of each bin (except last bin)
                        chi2 = chi2_contingency(bin(i), bin(1+1)) i: idx of bin
                 3. find the min chi2 and its idx: i
                 4. merge the bin i and bin i + 1
                 5. repeat step 2 to step 4 end when conditions satisfied
                 ( when the min chi-delta is too small or the n-bins is just as the input parameter)

    parameters
    ----------
    estimator : prebinning discretizer instance,
            estimator used to prebin the data
    n_bins : int, bin num
    n_jobs: int, default = None
            applicable when multi-threading
    chi2-square score:
    X^2 =Z∑(A_ij-E_ij )^2 /E_ij
    where E_ij= (N_i / N) * C_j. N is total count of merged bucket, N_i is the total count of ith bucket and C_i is the count of jth label in merged bucket. A_ij is number of jth label in ith bucket.
    
    References
    ----------
    [1] Shu, z. , and D Chen. "Discretization of continuous attributes based on Chi-square statistic and information entropy of rough sets." Journal of Zhejiang University 39.6(2005).
    """
    
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_bins=5, n_jobs=None, threshold=0.001):
        super(ChiMergeDiscretizer, self).__init__(n_bins=n_bins, n_jobs=n_jobs)
        self.estimator = estimator
        self.threshold = threshold

    def _validate_params(self):
        if not isinstance(self.estimator, BaseDiscretizer):
            raise ValueError("estimator should be of type 'BaseDiscretizer'.")
    
    def fit(self, X, y=None, **fit_params):
        """Fit the binner.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples,)
            Data to be discretized.
        y : numeric array-like, shape (n_samples,)
            best-ks is supervised estimator, and thus 'y' must be introduced

        Returns
        ----------
        self
        """
        self._validate_params()
        self.estimator.fit(X, y=y, **fit_params)
        return super(ChiMergeDiscretizer, self).fit(X, y=y)
    
    @property
    def closed(self):
        return self.estimator.closed

    @closed.setter
    def closed(self, value):
        self.estimator.closed = value

    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        pre_bin_edges = self.estimator.bin_edges_[i]
        n_pre_bins = len(pre_bin_edges) - 1
        n_bin = min(n_pre_bins, n_bin)

        threshold = self.threshold
        right = self.closed == 'right'
        indices = np.digitize(x, pre_bin_edges[1:-1], right=right)

        n_nonevents = np.zeros(n_pre_bins, dtype=np.int64)
        n_events = np.zeros(n_pre_bins, dtype=np.int64)

        for i in range(n_pre_bins):
            mask = (indices == i)
            bin_size = np.count_nonzero(mask)
            n_event = np.count_nonzero((y == 1) & mask)
            n_events[i] = n_event
            n_nonevents[i] = bin_size - n_event

        ind_mask = np.arange(len(pre_bin_edges))
        support_ = np.ones(len(pre_bin_edges), dtype=bool)
        curr_n_bins = np.count_nonzero(support_) - 1

        while curr_n_bins > n_bin:
            selected_ind = ind_mask[support_]
            events = np.add.reduceat(n_events, selected_ind[:-1])
            nonevents = np.add.reduceat(n_nonevents, selected_ind[:-1])
            chi2s = np.asarray([chi2_contingency([[e1, ne1], [e2, ne2]])[0] for e1, e2, ne1, ne2 in
                                zip(events[:-1], events[1:], nonevents[:-1], nonevents[1:])])
            min_chi2 = np.min(chi2s)
            if min_chi2 >= threshold:
                break
            indices = chi2s == min_chi2
            support_indices = selected_ind[1:-1][indices]
            support_[support_indices] = False
            curr_n_bins = np.count_nonzero(support_) - 1
        
        bin_edge = pre_bin_edges[support_]
        return bin_edge, curr_n_bins

    def _more_tags(self):
        return {
            "X_types": _safe_tags(self.estimator, "X_types"),
            "allow_nan": _safe_tags(self.estimator, "allow_nan"),
            "requires_y": True,
        }
