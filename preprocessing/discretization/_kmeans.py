import warnings
import numpy as np
from sklearn.utils import _deprecate_positional_args
from ._base import BaseDiscretizer, BaseShrinkByInflectionDiscretizer
from ._base import _MIN_BIN_WIDTH, _ATOL, _RTOL
from ...base import clone
from ...cluster import KMeansParameterProxy


class KMeansDiscretizer (KMeansParameterProxy, BaseDiscretizer):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None, init='uniform', n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=None, copy_x=True, algorithm='auto'):
        BaseDiscretizer.__init__(self, n_bins=n_bins, n_jobs=n_jobs)
        KMeansParameterProxy.__init__(self, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)

    def fit(self, X, y=None, **fit_params):
        self._make_estimator()
        return super(KMeansDiscretizer, self).fit(X, y=y, **fit_params)
    
    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        col_min, col_max = x.min(), x.max()

        if col_min == col_max:
            warnings.warn("Feature {} is constant and will be replaced with 0.".format(i))

            bin_edge = np.array([-np.inf, np.inf])
            n_bin = 1
            return bin_edge, n_bin
        
        # Deterministic initialization with uniform spacing
        if self.init == 'uniform':
            uniform_edges = np.linspace(col_min, col_max, n_bin + 1)
            init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
        else:
            init = self.init
        
        # 1D k-means procedure
        km = clone(self.estimator)
        km.n_clusters = n_bin
        km.init = init
        centers = km.fit(x[:, None]).cluster_centers_[:, 0]
        # Must sort, centers may be unsorted even with sorted init
        centers.sort()
        bin_edge = (centers[1:] + centers[:-1]) * 0.5
        bin_edge = np.r_[col_min, bin_edge, col_max]

        # Remove bins whose width are too smatl (i.e., <= le-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
        bin_edge = bin_edge[mask]

        if len(bin_edge) - 1 != n_bin:
            warnings.warn('Bins whose width are too small (i.e.,<=1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
            n_bin = len(bin_edge) - 1
        return bin_edge, n_bin


class KMeansDiscretizerShrinkByInflection(KMeansParameterProxy, BaseShrinkByInflectionDiscretizer):
    def __init__(self, *, n_bins=5, n_jobs=None, n_inflections=None, init='uniform', n_init=1, max_iter=300, tol=1e-4, verbose=0, random_state=None, copy_x=True, algorithm='auto'):
        BaseShrinkByInflectionDiscretizer._init_(self, n_bins=n_bins, n_jobs=n_jobs, n_inflections=n_inflections)
        KMeansParameterProxy.__init__(self, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        
    def fit(self, X, y=None, **fit_params):
        self._make_estimator()
        return super(KMeansDiscretizerShrinkByInflection, self).fit(X, y=y, **fit_params)
    
    def _bin_one_column(self, i, n_bin, x, y=None, n_inflection=None):
        col_min, col_max = x.min(), x.max()
        if col_min == col_max:
            warnings.warn("Feature {} is constant and will be replaced with 0.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            n_bin = 1
            return bin_edge, n_bin
        
        if self.closed == 'right':
            right = True
        else:
            right = False
        
        eps = _ATOL + _RTOL * np.abs(x)

        bad_mask, good_mask = np.equal(y, 1), np.not_equal(y, 1)
        bad_tot, good_tot = np.count_nonzero(bad_mask) + 2, np.count_nonzero(good_mask) + 2
        residue = np.log(good_tot / bad_tot)

        while True:
            # Deterministic initialization with uniform spacing
            if self.init == 'uniform':
                uniform_edges = np.linspace(col_min, col_max, n_bin + 1)
                init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
            else:
                init = self.init
            
            # 1D k-means procedure
            km = clone(self.estimator)
            km.n_clusters = n_bin
            km.init = init
            centers = km.fit(x[:, None]).cluster_centers_[:, 0]
            # Must sort, centers may be unsorted even with sorted init
            centers.sort()
            bin_edge = (centers[1:] + centers[:-1]) * 0.5
            bin_edge = np.r_[col_min, bin_edge, col_max]

            # Remove bins whose width are too small (i.e., <= 1e-8)
            mask = np.ediff1d(bin_edge, to_begin=np.inf) >_MIN_BIN_WIDTH
            bin_edge = bin_edge[mask]

            if len(bin_edge) - 1 != n_bin:
                warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. consider decreasing the number of bins.'.format(i))
                n_bin = len(bin_edge)-1
            
            if n_bin <= 2:
                break

            # Values wrich are close to a bin edge are susceptible to numeric instability. Add eps to X so these values are binned correctly
            # with respect to their decimal truncation. See documentation of numpy.isclose for an explanation of rtol and atol.
            xt = np.digitize(x + eps, bin_edge[1:], right=right)
            np.clip(xt, 0, n_bin - 1, out=xt)

            nums = np.zeros(n_bin, dtype=np.float64)
            for j in range (n_bin):
                bin_mask = np.equal(xt, j)
                # ignore unique values, this helps to prevent overfitting on id-Like coLumns .
                if np.count_nonzero(mask) == 1:
                    nums[j] = 0.
                else:
                    bad_num = np.count_nonzero(np.logical_and(bin_mask, bad_mask)) + 1
                    good_num = np.count_nonzero(np.logical_and(bin_mask, good_mask)) + 1
                    nums[j] = np.log(bad_num / good_num) + residue
            diffs = np.ediff1d(nums)
            curr_inflection = np.count_nonzero(diffs[1:] * diffs[:-1] < 0)
            if curr_inflection <= n_inflection:
                break
            n_bin -= 1
        return bin_edge, n_bin
