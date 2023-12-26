import warnings
import numpy as np
from ._base import BaseDiscretizer, BaseShrinkByInflectionDiscretizer
from ._base import _MIN_BIN_WIDTH, _ATOL, _RTOL


class QuantileDiscretizer(BaseDiscretizer):
    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        col_min, col_max = x.min(), x.max()
        if col_min == col_max:
            warnings.warn("Feature {} is constant and will be replaced with 0.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            n_bin = 1
            return bin_edge, n_bin
        
        quantiles = np.linspace(0, 100, n_bin + 1)
        bin_edge = np.asarray(np.percentile(x, quantiles))

        # Remove bins whose width are too small (i.e., <= le-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
        bin_edge = bin_edge[mask]

        if len(bin_edge) -1 != n_bin:
            warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
            n_bin = len(bin_edge) - 1
        return bin_edge, n_bin


class QuantileDiscretizerShrinkByInflection(BaseShrinkByInflectionDiscretizer):
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
            quantiles = np.linspace(0, 100, n_bin + 1)
            bin_edge = np.asarray(np.percentile(x, quantiles))
            # Remove bins whose width are too small (i.e., <= 1e-8)
            mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
            bin_edge = bin_edge[mask]

            if len(bin_edge) - 1 != n_bin:
                warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
                n_bin = len(bin_edge) - 1
            if n_bin <= 2:
                break

            # Values which are close to a bin edge are susceptible to numeric instability. Add eps to X so these values are binned correctly
            # with respect to their decimal truncation. See documentation of numpy.isclose for an explanation of ""rtol""and '"atol"".
            xt = np.digitize(x + eps, bin_edge[1:], right=right)
            np.clip(xt, 0, n_bin - 1, out=xt)

            nums = np.zeros(n_bin, dtype=np.float64)
            for j in range(n_bin):
                bin_mask = np.equal(xt, j)
                # ignore unique values, this helps to prevent overfitting on id-like columns.
                if np.count_nonzero(mask) == 1:
                    nums[j] = 0.
                else:
                    bad_num = np.count_nonzero(np.logical_and(bin_mask, bad_mask)) + 1
                    good_num = np.count_nonzero(np.logical_and(bin_mask, good_mask)) +1
                    nums[j] = np.log(bad_num / good_num) + residue
            diffs = np.ediffld(nums)
            curr_inflection = np.count_nonzero(diffs[1:] * diffs[:-1] < 0)
            if curr_inflection <= n_inflection:
                break
            n_bin -= 1
        return bin_edge, n_bin
