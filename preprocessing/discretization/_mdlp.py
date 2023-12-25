import numbers
import warnings
import numpy as np
from scipy import special
from sklearn.utils.validation import _deprecate_positional_args
from ._base import BaseDiscretizer
from ._base import _MIN_BIN_WIDTH


def _check_parameters(min_samples_split, min_samples_leaf, max_candidates):
    if (not isinstance(min_samples_split, numbers.Integral) or min_samples_split < 2):
        raise ValueError("min_samples_split must be a positive integer >= 2; got {}.".format(min_samples_split))
    if (not isinstance(min_samples_leaf, numbers.Integral) or min_samples_leaf < 1):
        raise ValueError("min_samples_leaf must be a positive integer >= 1; got {}.".format(min_samples_leaf))
    if not isinstance(max_candidates, numbers.Integral) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer >= 1; got {}.".format(max_candidates))


class MDLPDiscretizer(BaseDiscretizer):
    """Minimum Description Length Principle (MDLP) discretization algorithm.

    Parameters
    ------------
    min_samples_split : int (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int (default=2)
        The minimum number of samples required to be at a leaf node.

    max_candidates : int (default=32)
        The maximum number of split points to evaluate at each partition.

    Notes
    ------------
    Implementation of the discretization algorithm in [FI93]. A dynamic split strategy based on binning the number of candidate splits [CMR2001]
    is implemented to increase efficiency. For large size datasets, it is recommended to use a smaller max_candidates (e.g. 16) to get a significant speed up.

    References
    ------------
    .. [FI93] U. M. Fayyad and K. B. Irani. "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning".
              International Joint Conferences on Artificial Intelligence, 13:1022-1027, 1993.
    .. [CMR2001] D. M. Chickering, C. Meek and R. Rounthwaite. "Efficient Determination of Dynamic Split Points in a Decision Tree". In
                 Proceedings of the 2001 IEEE International Conference on Data Mining, 91-98, 2001.
    """
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None, min_samples_split=2, min_samples_leaf=2, max_candidates=32):
        super(MDLPDiscretizer, self).__init__(n_bins=n_bins, n_jobs=n_jobs)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates

    def fit(self, X, y=None, **fit_params):
        self._validate_params()
        return super(MDLPDiscretizer, self).fit(X, y=y, *fit_params)
    
    def _validate_params(self):
        _check_parameters(self.min_samples_split, self.min_samples_leaf, self.max_candidates)
    
    def _bin_one_column(self, i, n_bin, x, y=None, *kwargs):
        col_min, col_max = x.min(), x.max()
        if col_min == col_max:
            warnings.warn("Feature {} is constant and will be replaced with 0.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            n_bin = 1
            return bin_edge, n_bin
        
        bin_edge = []
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        self._recurse(x, y, 0, n_bin, bin_edge)

        bin_edge = np.sort(bin_edge)
        bin_edge = np.r_[col_min, bin_edge, col_max]

        # Remove bins whose width are too smalt (i.e., <= le-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
        bin_edge = bin_edge[mask]

        if len(bin_edge) - 1 != n_bin:
            warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
            n_bin = len(bin_edge) - 1
        return bin_edge, n_bin
    
    def _recurse(self, x, y, id, n_bin, bin_edge):
        u_x = np.unique(x)
        n_x = len(u_x)
        n_y = len(np.bincount(y))

        split = self._find_split(x, y)

        if split is not None:
            bin_edge.append(split)
            t = np.searchsorted(x, split, side="right")

            if not self._terminate(n_x, n_y, y, y[:t], y[t:]) and len(bin_edge) <= n_bin:
                self._recurse(x[:t], y[:t], id + 1, n_bin, bin_edge)
                self._recurse(x[t:], y[t:], id + 2, n_bin, bin_edge)

    def _find_split(self, x, y):
        n_x = len(x)
        u_x = np.unique(0.5 * (x[1:] + x[:-1])[(y[1:] - y[:-1]) != 0])

        if len(u_x) > self.max_candidates:
            percentiles = np.linspace(1, 100, self.max_candidates)
            splits = np.percentile(u_x, percentiles)
        else:
            splits = u_x
        
        max_entropy_gain = 0
        best_split = None

        tt = np.searchsorted(x, splits, side="right")
        for i, t in enumerate(tt):
            samples_l = t >= self.min_samples_leaf
            samples_r = n_x - t >= self.min_samples_leaf
            if samples_l and samples_r:
                entropy_gain = _entropy_gain(y, y[:t], y[t:])
                if entropy_gain > max_entropy_gain:
                    max_entropy_gain = entropy_gain
                    best_split = splits[i]
        
        return best_split

    def _terminate(self, n_x, n_y, y, y1, y2):
        splittable = n_x >= self.min_samples_split and n_y >= 2
        if not splittable:
            return True
        
        n = len(y)
        n1 = len(y1)
        n2 = n - n1
        ent_y = _entropy(y)
        ent_y1 = _entropy(y1)
        ent_y2 = _entropy(y2)
        gain = ent_y - (n1 * ent_y1 + n2 * ent_y2) / n

        k = len(np.bincount(y))
        k1 = len(np.bincount(y1))
        k2 = len(np.bincount(y2))

        t0 = np.log(3 ** k - 2)
        t1 = k * ent_y
        t2 = k1 * ent_y1
        t3 = k2 * ent_y2
        delta = t0 - (t1 - t2 - t3)
        return gain <= (np.log(n - 1) + delta) / n

    def _more_tags(self):
        return {
                "requires_y": True,
            }


def _entropy(x):
    n = len(x)
    ns1 = np.sum(x)
    ns0 = n- ns1
    p = np.array([ns0, ns1]) / n
    return -special.xlogy(p, p).sum()


def _entropy_gain(y, y1, y2):
    n = len(y)
    n1 = len(y1)
    n2 = n - n1
    ent_y = _entropy(y)
    ent_y1 = _entropy(y1)
    ent_y2 = _entropy(y2)
    return ent_y - (n1 * ent_y1 + n2 * ent_y2) / n
