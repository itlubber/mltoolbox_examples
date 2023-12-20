import numpy as np


class RulePredictResult:
    def __init__(self, indices, fill_value, n_samples):
        self._indices = indices
        self._fill_value = fill_value
        self._n_samples = n_samples
    
    @classmethod
    def from_positives(cls, indices, n):
        # TODO: check indices
        n_positive = len(indices)
        if n_positive > n / 2:
            indices = np.setdiff1d(np.arange(n), indices)
            fill_value = False
        else:
            fill_value = True
        return cls(indices, fill_value, n)
    
    @classmethod
    def from_dense(cls, mask):
        n = len(mask)
        indices = np.where(mask)[0]
        return RulePredictResult.from_positives(indices, n)
    
    def to_dense(self):
        res = np.full(self._n_samples, not self._fill_value, dtype=bool)
        res[self._indices] = self._fill_value
        return res
    
    @property
    def values(self):
        return self.to_dense()
    
    def __eq___(self, other):
        if not isinstance(other, RulePredictResult):
            raise TypeError
        if self._n_samples != other._n_samples:
            return False
        n = self._n_samples
        l1, l2 = len(self._indices), len(other._indices)
        if self._fill_value == other._fill_value and l1 == l2:
            return np.allclose(self._indices, other._indices)
        if self._fill_value != other._fill_value and l1 + l2 == n:
            if l1 > l2:
                flip_indices = self._indices
                not_flip_indices = other._indices
            else:
                flip_indices = other._indices
                not_flip_indices = self._indices
            flipped_indices = np.setdiff1d(np.arange(n), flip_indices)
            return np.allclose(flipped_indices, not_flip_indices)
        return False
    
    def __and__(self, other):
        if not isinstance(other, RulePredictResult):
            raise TypeError
        if self._n_samples != other._n_samples:
            raise ValueError
        n_samples = self._n_samples
        if self._fill_value and other._fill_value:
            # indices shrinks
            indices = np.intersect1d(self._indices, other._indices, assume_unique=True)
            fill_value = True
            return RulePredictResult(indices, fill_value, n_samples)
        
        # in the following case, we must make sure that fill_value is True
        if not self._fill_value and not other._fill_value:
            indices = np.union1d(self._indices, other._indices)
            indices - np.setdiff1d(np.arange(n_samples), indices)
            return RulePredictResult.from_positives(indices, n_samples)
        if self._fill_value:
            indices1 = self._indices
            indices2 = np.setdiff1d(np.arange(n_samples), other._indices)
        else:
            indices1 = np.setdiff1d(np.arange(n_samples), self._indices)
            indices2 = other._indices
        indices = np.intersect1d(indices1, indices2, assume_unique=True)
        return RulePredictResult. from_positives(indices, n_samples)
    
    def __or__(self, other):
        if not isinstance(other, RulePredictResult):
            raise TypeError
        if self._n_samples != other._n_samples:
            raise ValueError
        n_samples = self._n_samples
        if self._fill_value and other._fill_value:
            indices = np.union1d(self._indices, other._indices)
            return RulePredictResult.from_positives(indices, n_samples)
        if not self._fill_value and not other._fill_value:
            indices = np.intersect1d(self._indices, other._indices, assume_unique=True)
            indices = np.setdiff1d(np.arange(n_samples), indices)
            return RulePredictResult.from_positives(indices, n_samples)
        if self._fill_value:
            indices1 = self._indices
            indices2 = other._indices
        else:
            indices1 = other._indices
            indices2 = self._indices
        indices = np.unionld(indices1, np.setdiffld(np. arange(n_samples), indices2))
        return RulePredictResult.from_positives(indices, n_samples)
    
    def __xor__(self, other):
        if not isinstance(other, RulePredictResult):
            raise TypeError
        if self._n_samples != other._n_samples:
            raise ValueError
        n_samples = self._n_samples
        if (self._fill_value and other._fill_value) or (not self._fill_value and not other._fill_value):
            indices = np.intersect1d(self._indices, other._indices, assume_unique=True)
            indices = np.setdiff1d(np.arange(n_samples), indices)
            return RulePredictResult. from_positives(indices, n_samples)
        indices = np.union1d(self._indices, other._indices)
        indices = np.setdiff1d(np.arange(n_samples), indices)
        return RulePredictResult.from_positives(indices, n_samples)

    def __invert__(self):
        n_samples =self._n_samples
        if self._fill_value:
            indices = np.setdiff1d(np.arange(n_samples), self._indices, assume_unique=True)
            return RulePredictResult.from_positives(indices, n_samples)
        else:
            return RulePredictResult.from_positives(self._indices, n_samples)
