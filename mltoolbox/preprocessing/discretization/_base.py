import numbers
import warnings
from abc import abstractmethod

import numpy as np
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_array, check_is_fitted, FLOAT_DTYPES, _num_samples, _deprecate_positional_args
from ._discretization import KBinsDiscretizer
from ...base import BaseEstimator, TransformerMixin

_MIN_BIN_WIDTH = 1e-8
_RTOL = 1e-5
_ATOL = 1e-8


def _digitize_one_column(i, x, bin_edge, n_bin, right):
    """Digitize cut one feature column into bin indices

    Parameters
    ----------
    i: int
        Feature index.
    x : array-like of shape (n_samples, )
        Single feature column.
    bin_edge : array-like of shape (n_bins + 1,)
        Split points for this feature.
    n_bin : int
        Number of bins.
    right : bool
        Whether the bin edge is closed on right.

    Returns
    ----------
    xt : array-like of shape (n_samples, 1)
        New column representing bin indices.
    """
    # Values which are close to a bin edge are susceptible to numeric
    # instability. Add eps to X so these values are binned correctly
    # with respect to their decimal truncation. See documentation of
    # numpy.isclose for an explanation of '^rtol"" and ^atol^"
    # eps =_ATOL +_RTOL*np.abs(x)
    # xt = np.digitize(x + eps, bin_edge[1:], right=right)
    xt = np.digitize(x, bin_edge[1:], right=right)
    np.clip(xt, 0, n_bin - 1, out=xt)
    # xt[np.isnan(x)] = -1
    return xt[:, np.newaxis]


class BaseDiscretizer(TransformerMixin, BaseEstimator):
    """Bin continuous data into intervals.

    Read more in the :ref: 'User Guide <preprocessing_discretization>.

    Parameters
    ----------
    n_bins : int or array-like, shape (n_features,) (default=5)
        The number of bins to produce. Raises ValueError if n_bins < 2

    Attributes
    ----------
    n_bins_ : int array, shape (n_features,)
        Number of bins per feature. Bins whose width are too small (i.e., <= 1e-8) are removed with a warning.
    bin_edges_: list of arrays, shape (n_features, )
        The edges of each bin. Contain arrays of varying shapes (n_bins_,). Ignored features will have empty arrays.

    See Also
    ----------
    mltoolbox.preprocessing.Binarizer : Class used to bin values as `0` or `1` based on a parameter `threshold`
    
    Notes
    ----------
    In bin edges for feature `i`, the first and last values are used only for
    `inverse_transform`. During transform, bin edges are extended to::

    пр.concatenate([-np.inf, bin_edges_[i][1:-1], np.inf])

    For example, you can combine `DecisionTreeDiscretizer` with
    :class:`orca_m1.compose.ColumnTransformer` if you only want to preprocess part of the features.
    DecisionTreeDiscretizer might produce constant features (e.g., when
    data is encoded using one-hot, and certain bins do not contain any data).
    These features can be removed with feature selection algorithms
    (e.g., :class:`mltoolbox.feature_selection.VarianceThreshold`).

    Examples
    ----------
    >>> from mltoolbox.preprocessing import DecisionTreeDiscretizer
    >>>X = [[-2,1,-4,-1],
            [-1, 2, -3, -0.5],
            [0,  3, -2,  0.5],
            [1,  4, -1,    2]]
    >>> est = DecisionTreeDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    >>> est.fit(X)
    DecisionTreeDiscretizer(...)
    >>> Xt = est.transform(x)
    >>> Xt # doctest: +SKIP
    array(  [( 0., 0., 0., 0.],
            [ 1., 1., 1., 0.J,
            [ 2., 2., 2., 1.],
            [ 2., 2., 2., 2.]])

    Sometimes it may be useful to convert the data back into the original
    feature space. The `inverse_transform` function converts the binned
    data into the original feature space. Each value will be equal to the mean
    of the two bin edges.

    >>> est.bin_edges_[0]
    array([-2., -1., 0., 1.])
    >>> est.inverse_transform(Xt)
    array(  [[-1.5, 1.5, -3.5, -0.5],
            [-0.5, 2.5, -2.5, -0.5],
            ( 0.5, 3.5, -1.5, 0.5],
            [ 0.5, 3.5, -1.5, 1.5]])
    """
    _closed = "left"

    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None):
        self.n_bins = n_bins
        self.n_jobs = n_jobs
    
    @property
    def closed(self):
        return self._closed
    
    @closed.setter
    def closed(self, value):
        if value not in ("left", "right"):
            raise ValueError("`closed` should be 'left' or 'right'!")
        self._closed = value

    @closed.deleter
    def closed(self):
        AttributeError("Can not delete this attribute!")

    def fit(self, X, y=None, **fit_params):
        """Fit the estimator.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with
        :class:`sklearn.pipeline.Pipeline`.

        Returns
        ----------
        self
        """
        n_bins = self._validate_n_bins(X)
        X = check_array(X, dtype="numeric")
        # X = check_array(X, dtype="numeric", force_all_finite=False)
        if _safe_tags(self, key='requires_y'):
            if y is None:
                raise ValueError("Supervised discretization requires y.")
            y = self._check_y(y)

        _, n_features = X.shape
        result = Parallel(n_jobs=self.n_jobs)(delayed(self._bin_one_column)(i, n_bin, X[:, i], y) for i, n_bin in enumerate(n_bins))
        bin_edges_, n_bins_ = zip(*result)
        # bin_edges_ contains different Length of bin_edges, numpy array should use type object
        # self.bin_edges_ = np.asarray(bin_edges_, dtype=object)
        self.bin_edges_ = list(bin_edges_)
        self.n_bins_ = np.asarray(n_bins_, dtype=int)
        return self
    
    def _validate_n_bins(self, X):
        """Returns n_bins_, the number of bins per feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        if isinstance(X, list) and len(X) > 0:
            n_features = len(X[0])
        else:
            _, n_features = X.shape
        orig_bins = self.n_bins
        if orig_bins is None:
            raise ValueError("'n_bins' should not be empty!")

        if isinstance(orig_bins, numbers.Number):
            if not isinstance(orig_bins, numbers.Integral):
                raise ValueError("{} received an invalid n_bins type. Received {}, expected int.".format(self.__class__.__name__, type(orig_bins).__name__))
            if orig_bins < 2:
                raise ValueError("{} received an invalid number of bins. Received {}, expected at least 2.".format(self.__class__.__name__, orig_bins))
            return np.full(n_features, orig_bins, dtype=int)
    
        if isinstance(orig_bins, dict):
            if isinstance(X, DataFrame):
                n_bins = []
                for f in X.columns:
                    if f not in orig_bins:
                        warnings.warn("No number of bins for feature {}!".format(f))
                    n_bin = orig_bins.get(f, 5)
                    if not isinstance(n_bin, numbers.Integral):
                        raise ValueError("{} received an invalid number of bins  type. Received {}, expected int.".format(self.__class__.__name__, type(n_bin).__name__))
                    if n_bin <2:
                        raise ValueError("{} received an invalid number of bins. Received {}, expected at least 2.".format(self.__class__.__name__, n_bin))
                    n_bins.append(n_bin)
                return np.asarray(n_bins, dtype=int)
            else:
                raise ValueError("Number of bins should be a dictionary only for dataframe input!")
            
        n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)

        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array of shape (n_features,).")
        
        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid number of bins at indices {}. Number of bins must be at least 2, and must be an int.".format(self.__class__.__name__, indices))
        
        return n_bins

    def _check_y(self, y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        return y
    
    def get_splits(self, i):
        """Get the split points of the i-th feature.

        Parameters
        ----------
        i : int
            Column index

        Returns
        ----------
        splits : np.ndarray
            The split points
        """
        check_is_fitted(self, 'bin_edges_')
        return self.bin_edges_[i][1:-1]
    
    @property
    def splits_(self):
        """Get all split points.

        Returns
        ----------
        splits : list
            List of split points.
        """
        check_is_fitted(self, 'bin_edges_')
        bin_edges = self.bin_edges_
        return [bin_edges[1:-1] for bin_edges in bin_edges]
    
    @abstractmethod
    def _bin_one_column(self, i, n_bin, X, y=None, **kwargs):
        pass

    def _transform(self, x):
        """Discretize the data.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
        Data to be discretized.

        Returns
        ----------
        Xt : numeric array-like or sparse matrix
            Data in the binned space.
        """
        check_is_fitted(self)

        # force copy to avoid modifying original data
        X = check_array(x, copy=True, dtype=FLOAT_DTYPES)

        # X = check_array(X, copy=False, dtype=FLOAT_DTYPES, force_all_finite=False)
        n_features = self.n_bins_.shape[0]
        if X.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, received {}.".format(n_features, x.shape[1]))
        
        n_bins = self.n_bins
        bin_edges = self.bin_edges_
        right = self.closed == 'right'
        # for jj in range(n_features):
        #     # Values which are close to a bin edge are susceptible to numeric
        #     # instability. Add eps to x so these values are binned correctly
        #     # with respect to their decimal truncation. See documentation of
        #     # numpy.isclose for an explanation of rtol and atol.
        # eps = _ATOL + _RTOL * np.abs(Xt[:, jj])
        # Xt[:, jj] = np.digitize(Xt[:, jj] + eps, bin_edges[jj][1:], right=right)
        # np.clip(Xt, e, self.n_bins_ - 1, out=Xt)
        xt_list = Parallel(n_jobs=self.n_jobs)(delayed(_digitize_one_column)(i, X[:, i], bin_edges[i], n_bins[i], right) for i in range(n_features))
        Xt = np.hstack(xt_list)
        return Xt
        
    def transform(self, X):
        data = self._transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, columns=columns, index=index).astype(dtype='category')
        return data
    
    def _inverse_transform(self, Xt):
        """Transform discretized data back to original feature space

        Note that this function does not regenerate the original data due to discretization rounding.

        Parameters
        ----------
        Xt : numeric array-like, shape (n_sample, n_features)
            Transformed data in the binned space.

        Returns
        ----------
        Xinv : numeric array-like
            Data in the original feature space.
        """
        check_is_fitted(self)
        Xinv = check_array(Xt, copу=True, dtype=FLOAT_DTYPES)
        n_features = self.n_bins_.shape[0]
        if Xinv.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, received {}.".format(n_features, Xinv.shape[1]))
        
        for jj in range(n_features):
            bin_edges = self.bin_edges_[jj]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            Xinv[:, jj] = bin_centers[int(Xinv[:, jj])]

        return Xinv
    
    def inverse_transform(self, Xt):
        data = self._inverse_transform(Xt)
        if isinstance(Xt, DataFrame):
            columns = Xt.columns
            index = Xt.index
            return DataFrame (data=data, columns=columns, index=index)
        return data
        
    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_y": False,
        }


class BaseMergeByMinSamplesBinDiscretizer(BaseDiscretizer):
    def _init_(self, *, n_bins=5, n_jobs=None, min_samples_bin=0.05):
        super(BaseMergeByMinSamplesBinDiscretizer, self)._init_(n_bins=n_bins, n_jobs=n_jobs)
        self.min_samples_bin = min_samples_bin

    def _validate_min_samples_bin(self, X):
        min_samples_bin = self.min_samples_bin
        n_samples = _num_samples(X)
        if not isinstance(min_samples_bin, numbers.Number):
            raise ValueError("{} received an invalid min_samples_bin type. Received {}, expected a real number.".format(self.__class__.__name__, type(min_samples_bin).__name__))
        
        if isinstance(min_samples_bin, float):
            if not 0.0 < min_samples_bin < 1.0:
                raise ValueError("{} received an invalid number of min_samples_bin. Received {}, expect value between 0 and 1.".format(self.__class__.__name__, min_samples_bin))
            min_samples_bin_ = int(min_samples_bin * n_samples)
        elif isinstance(min_samples_bin, numbers.Integral):
            if not 0 < min_samples_bin < n_samples:
                raise ValueError("{} received an invalid number of min_samples_bin. Received {}, expect value between 0 and {}.".format(self.__class__.__name__, min_samples_bin, n_samples))
            min_samples_bin_ = min_samples_bin
        else:
            raise ValueError ("{} received an invalid min_samples_bin type. Received {}, expected int or float between 0 and 1.".format(self.__class__.__name__, type(min_samples_bin).__name__))
        return min_samples_bin_
    
    @abstractmethod
    def _bin_one_column(self, i, n_bin, x, y=None, min_samples_bin=None):
        pass
    
    def fit(self, x, y=None, **fit_params):
        """Fit the estimator.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with :class: `sklearn.pipeline.Pipeline`.

        Returns 
        ----------
        self
        """
        n_bins = self._validate_n_bins(X)
        min_samples_bin = self._validate_min_samples_bin(X)
        X = check_array(X, dtype="numeric")
        if _safe_tags(self, key='requires_y'):
            if y is None:
                raise ValueError("Supervised discretization requires y.")
            y = self._check_y(y)
        
        _, n_features = X.shape
        result = Parallel(n_jobs=self.n_jobs)(delayed(self._bin_one_column)(i, X[:, i], n_bin, y, min_samples_bin=min_samples_bin) for i, n_bin in zip(range(n_features), n_bins))
        bin_edges_, n_bins_= zip(*result)
        self.bin_edges_ = np.asarray(bin_edges_, dtype=object)
        self.n_bins_ = np.asarray(n_bins_, dtype=int)
        return self


class BaseShrinkByInflectionDiscretizer(BaseDiscretizer):
    def __init_(self, *, n_bins=5, n_jobs=None, n_inflections=1):
        super(BaseShrinkByInflectionDiscretizer, self)._init_(n_bins=n_bins, n_jobs=n_jobs)
        self.n_inflections = n_inflections

    def _validate_n_inflections(self, X):
        _, n_features = X.shape
        inflections = self.n_inflections
        if inflections is None:
            raise ValueError("'inflections' should not be empty!")
        if isinstance(inflections, numbers.Number):
            if not isinstance(inflections, numbers.Integral):
                raise ValueError("{} received an invalid number of inflections. Received {}, expected a integer.".format(self.__class__.__name__, inflections))
            if inflections < 0:
                raise ValueError("{} received an invalid number of inflections. Received {}, expected at least ?.".format(self.__class__.__name__, inflections))
            return np.full(n_features, inflections, dtype-int)
    
        if isinstance(inflections, dict):
            if isinstance(X, DataFrame):
                inflections_ = []
                for f in X.columns:
                    # if f not in inflections:
                    #     raise ValueError ("No inflection number for feature {}!".format(f))
                    inflection = inflections.get(f, 1)
                    if not isinstance(inflection, numbers.Integral):
                        raise ValueError("{} received an invalid inflection type. Received {}, expected int.".format(self.__class__.__name__, type(inflection).__name__))
                    if inflection < 0:
                        raise ValueError("{} received an invalid number of inflections. Received (}, expected at least 0.".format(self.__class__.__name__, inflection))
                    inflections_.append(inflection)
                inflections_ = np.asarray(inflections_, int)
            else:
                raise ValueError("Inflexions should be a dictionary only for dataframe input! ")
        else:
            inflections_ = np.asarray(inflections, dtype=int)
            if inflections_.size != X.shape[1]:
                raise ValueError("Inconsistent length for inflections and X!")
            for inflection in inflections_:
                if not isinstance(inflection, numbers.Integral):
                    raise ValueError("{} received an invalid inflection type. Received {}, expected int.".format(self.__class__.__name__, type(inflection).__name__))
            if inflection < 0:
                raise ValueError(" {} received an invalid number of inflections. Received {}, expected at least 0.".format(self.__class__.__name__, inflection))
        return inflections

    @abstractmethod
    def _bin_one_column(self, i, x, n_bin, y=None, n_inflection=None):
        pass

    def fit(self, X, y=None, **fit_params):
        """Fit the estimator

        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.
        y : None
            Ignored. This parameter exists only for compatibility with :class:`sklearn.pipeline.Pipeline`.

        Returns
        ----------
        self
        """
        n_bins = self._validate_n_bins(X)
        n_inflections = self._validate_n_inflections(X)
        X = check_array(X, dtype="numeric")
        if _safe_tags(self, key='requires_y'):
            if y is None:
                raise ValueError("Supervised discretization requires y.")
            y = self._check_y(y)

        _, n_features = X.shape
        result = Parallel(n_jobs=self.n_jobs)(delayed(self._bin_one_column)(i, n_bin, X[:, i], y, n_inflection=n_inflection) for i, n_bin, n_inflection in zip(range(n_features), n_bins, n_inflections))
        bin_edges_, n_bins_ = zip(*result)
        self.bin_edges_ = np.asarray(bin_edges_, dtype=object)
        self.n_bins_ = np.asarray(n_bins_, dtype=int)
        return self


def is_discretizer(obj):
    """Determine whether an object is discretizer.

    Parameters
    ----------
    obj : class or object
    
    Returns
    ----------
    truth : bool
        True if the object is subclass of 'BaseDiscretizer or an instance of 'BaseDiscretizer, else False
    """
    return issubclass(obj, BaseDiscretizer) or isinstance(obj, BaseDiscretizer)


def isa_discretizer(est):
    """Determine whether an object is an instance of discretizer.

    Parameters
    ----------
    est : object
        The obiject.

    Returns
    ----------
    truth : bool
        True if the object is an instance of BaseDiscretizer, else False
    """
    return isinstance(est, BaseDiscretizer)


def is_kbins_discretizer(obj):
    """Determine whether an object is discretizer.

    Parameters
    ----------
    obj : class or obiect
    
    Returns
    ----------
    truth : bool
        True if the object is subclass of `KBinsDiscretizer` or an instance of `KBinsDiscretizer`.
    """
    return issubclass(obj, KBinsDiscretizer) or isinstance (obj, KBinsDiscretizer)


def isa_kbins_discretizer(est):
    """Determine whether an object is an instance of discretizer.

    Parameters
    ----------
    est : object

    Returns
    ----------
    truth : bool
        True if the object if an instance of KBinsDiscretizer, else False.
    """
    return isinstance(est, KBinsDiscretizer)
