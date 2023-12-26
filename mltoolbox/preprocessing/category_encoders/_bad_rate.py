import numpy as np
from scipy.stats import rankdata

from . _base import BaseReplacer


def _check_binary_target(y):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(le.classes_) != 2:
        raise ValueError("Binary target is required.")
    return y


# TODO: accelerate this code
# TODO: fix bugs in inverse_transform
class BadRateEncoder(BaseReplacer):
    """
    Examples
    ---------
    >>> import pandas as pd
    >>> from mltoolbox.preprocessing import BadRateEncoder
    >>> X = pd.DataFrame({'b': ["上", "下", "左", "右", "右", "左"], 'c':['c', 'd', 'e', 'a', 'c','b']})
    >>> y = pd.Series([0, 1, 0, 1, 1, 1], name='y')
    >>> enc = BadRateEncoder(dtype=np.float64)
    >>> Xt = enc.fit_transform(X, y)
    >>> enc.inverse_transform(Xt)
    """
    def _fit_to_numbers(self, X, y):
        y = _check_binary_target(y)
        numbers_ = []
        dtype = self.dtype
        for i, cats in enumerate(self.categories_):
            col = self._get_feature(X, i)
            nums = np.zeros(cats.shape, dtype=dtype)
            for j, cat in enumerate(cats):
                mask = col == cat
                bad_num = np.count_nonzero(np.logical_and(mask, y == 1))
                tot_num = np.count_nonzero(mask)
                nums[j] = (bad_num / tot_num) if bad_num else 0.
            numbers_.append(nums)
        self.numbers = numbers_


class BadRateRankEncoder(BaseReplacer):
    """
    Examples
    ---------
    >>> import pandas as pd
    >>> from mltoolbox.preprocessing import BadRateRankEncoder
    >>> X = pd.DataFrame({'b': ["上", "下", "左", "右", "右", "左"], 'c':['c', 'd', 'e', 'a', 'c','b']})
    >>> y = pd.Series([0, 1, 0, 1, 1, 1], name='y')
    >>> enc = BadRateRankEncoder(dtype=np.float64)
    >>> Xt = enc.fit_transform(X,y)
    >>> enc.inverse_transform(xt)
    """
    def __init__(self, *, categories='auto', dtype=np.float64, handle_unknown='ignore', method='dense', invert=False):
        super(BadRateRankEncoder, self).__init__(categories=categories, dtype=dtype, handle_unknown=handle_unknown)
        self.method = method
        self.invert = invert

    def _fit_to_numbers(self, X, y):
        y = _check_binary_target(y)
        numbers_ = []
        dtype, method, invert = self.dtype, self.method, self.invert
        for i, cats in enumerate(self.categories_):
            col = self._get_feature(X, i)
            event_rates = np.zeros(cats.shape, dtype=np.float64)
            for j, cat in enumerate(cats):
                mask = col == cat
                n_event = np.count_nonzero(np.logical_and(y == 1, mask))
                bin_size = np.count_nonzero(mask)
                event_rates[j] = (n_event / bin_size) if n_event else 0.
            if invert:
                event_rates = -event_rates
            nums = rankdata(event_rates, method=method)
            # nums = rankdata(bad_rates, method=method).astype(dtype, сopy=False)
            numbers_.append(nums)
        self.numbers_ = numbers_


EventRateEncoder = BadRateEncoder
EventRateRankEncoder = BadRateRankEncoder
