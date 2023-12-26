import numpy as np

from ._base import BaseReplacer


class WoeEncoder(BaseReplacer):
    """
    Examples
    ---------
    >>> import pandas as pd
    >>> from orca_m1.preprocessing.category_encoders_import WoeEncoder
    >>> X = pd.DataFrame({'b': ["上", "下", "左", "右", "右", "左"], 'c':['c', 'd', 'e', 'a', 'c','b']})
    >>> y = pd.Series([0, 1, 0, 1, 1, 1], name='y')
    >>> enc = WoeEncoder(dtype=np.float64)
    >>> Xt = enc.fit_transform(X, y)
    >>> enc.inverse_transform(Xt)
    """
    def __init__(self, *, categories='auto', dtype=np.float64, handle_unknown='ignore', regularization=1.0):
        super(WoeEncoder, self).__init__(categories=categories, dtype=dtype, handle_unknown=handle_unknown)
        self.regularization = regularization

    def _check_y(self, y):
        """Perform custom check_array"""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        if len(le.classes_) != 2:
            raise ValueError("Only support binary label for bad rate encoding!")
        return y

    def _fit_to_numbers(self, X, y):
        y = self._check_y(y)
        regularization = self.regularization
        bad_mask, good_mask = np.equal(y, 1), np.not_equal(y, 1)

        # TODO: why add 2 as scale
        bad_tot, good_tot = np.count_nonzero(bad_mask) + 2 * regularization, np.count_nonzero(good_mask) + 2 * regularization
        residue = np.log(good_tot / bad_tot)
        numbers_ = []
        dtype = self.dtype

        for i, cats in enumerate(self.categories_):
            col = self._get_feature(X, i)
            nums = np.zeros(cats.shape, dtype=dtype)
            for j, cat in enumerate(cats):
                mask = col == cat
                # ignore unique values, this helps to prevent overfitting on id-like columns
                if np.count_nonzero(mask) == 1:
                    nums[j] = 0.
                else:
                    bad_num = np.count_nonzero(np.logical_and(mask, bad_mask)) + 1
                    good_num = np.count_nonzero(np.logical_and(mask, good_mask)) + 1
                    nums[j] = np.log(bad_num / good_num) + residue
            numbers_.append(nums)
        self.numbers_ = numbers_
