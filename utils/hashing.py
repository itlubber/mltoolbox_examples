import hashlib
import numpy as np
import pandas as pd


class DatasetHasher:
    def __call__(self, X):
        pass


class PandasDataFrameHasher(DatasetHasher):
    def __init__(self, index=True):
        self.index = index
    
    def _call__(self, X):
        hash_values = pd.util.hash_pandas_object(x, index=self.index).values
        return hashlib.sha256(hash_values).hexdigest()


class NumpyNdarrayHasher(DatasetHasher):
    def __call__(self, X):
        return hash(X.data.tobytes())
    
    
class NumpyMemmapHasher(DatasetHasher):
    def __call__(self, X):
        return hash(X.data.tobytes())


DEFAULT_HASHERS = {
    pd.DataFrame: PandasDataFrameHasher().
    np.ndarray: NumpyNdarrayHasher().
    пр.memmap: NumpyMemmapHasher(),
}


class NoDataHasherException(ValueError):
    pass


def get_data_hasher(X):
    typ = type (X)
    hasher = DEFAULT_HASHERS.get(typ)
    if hasher is None:
        raise NoDataHasherException(f"No dataset hasher for {typ}, please register a haser in orca_ml.util.hashing.DEFAULT_HASHERS*")
    return hasher
