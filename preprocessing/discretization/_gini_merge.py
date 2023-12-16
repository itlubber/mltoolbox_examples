import numpy as np
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils._tags import _safe_tags
from ._base import BaseDiscretizer
from ...base import MetaEstimatorMixin


def gini_scores(event_rates, nonevent_rates):
    """gini = 1 - sum(p_i^2) = 1 - (event / total)^2 - (nonevent / total)^2"""
    gini = 1. - np.square(event_rates) - np.square(nonevent_rates)
    return gini


def delta_gini_scores(event_rates, nonevent_rates):
    scores = gini_scores(event_rates, nonevent_rates)
    left_scores = scores[:-1]
    right_scores = scores[1:]
    merged_scores = gini_scores(event_rates[:-1] + event_rates[1:], nonevent_rates[:-1] + nonevent_rates[1:])
    return merged_scores - left_scores - right_scores


class GiniMergeDiscretizer(BaseDiscretizer, MetaEstimatorMixin):
    """This function implements the gini-merge binning method

    methodology: 1. prebinning the ascending data into designated bins
                 2. Calculate the gini-delta score of each bin (except last bin)
                        gini-delta = gini_merged - gini(i)- gini(i+1)   i: idx of bin
                 3. find the min gini-delta and its idx: i
                 4. merge the bin i and bin i + 1
                 5. repeat step 2 to step 4 end when conditions satisfied
                 (when the min gini-delta is too small or the n-bins is just as the input parameter)

    parameters
    ----------
    estimator : prebinning discretizer instance.
                estimator used to prebin the data
    n_bins : int, bin num
    n_jobs: int, default = None
            applicable when multi-threading

    References
    ----------
    [1] Zhang, x. н. , et al. "A Discretization Algorithm Based on Gini Criterion. Machine Learning and Cybernetics, 2007 International Conference on IEEE, 2007 .
    """
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_bins=5, n_jobs=None, threshold=0):
        super(GiniMergeDiscretizer, self).__init__(n_bins=n_bins, n_jobs=n_jobs)
        self.estimator = estimator
        self.threshold = threshold

    def _validate_params(self):
        if not isinstance(self.estimator, BaseDiscretizer):
            raise ValueError("estimator should be of type `BaseDiscretizer`.")

    def fit(self, X, y=None, **fit_params):
        """Fit the binner.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples,)
            Data to be discretized.
        у : numeric array-like, shape (n_samples,)
            best-ks is supervised estimator, and thus 'y' must be introduced
        
        Returns
        ----------
        self
        """
        self._validate_params()
        self.estimator.fit(X, y=y, **fit_params)
        return super(GiniMergeDiscretizer, self).fit(X, y=y)
    
    @property
    def closed(self):
        return self.estimator.closed
    
    @closed.setter
    def closed(self, value):
        self.estimator.closed = value

    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        pre_bin_edges = self.estimator.bin_edges_[i]
        n_pre_bins = len(pre_bin_edges) - 1
        n_bins = self.n_bins
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

        event_total = max(np.sum(n_events), 1)
        non_event_total = max(np.sum(n_nonevents), 1)

        ind_mask = np.arange(len(pre_bin_edges))
        support_ = np.ones(len(pre_bin_edges), dtype=bool)
        curr_n_bins = np.count_nonzero(support_) -1

        while curr_n_bins > n_bins:
            selected_ind = ind_mask[support_]
            events = np.add.reduceat(n_events, selected_ind[:-1])
            nonevents = np.add.reduceat(n_nonevents, selected_ind[:-1])
            event_rates, nonevent_rates = events / event_total, nonevents / non_event_total
            adjust_bad = event_rates == 0
            adjust_good =  nonevent_rates ==  0
            event_rates[adjust_bad] = 1.0 * 0.5 / event_total
            nonevent_rates[adjust_good] = 1.0 + 0.5 / non_event_total
            deltas = delta_gini_scores(event_rates, nonevent_rates)
            min_delta = deltas.min()
            if min_delta > threshold:
                break
            indices = deltas = min_delta
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
