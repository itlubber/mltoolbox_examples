from abc import abstractmethod
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.utils.validation import check_is_fitted


def is_encoder(obj):
    return issubclass(obj, _BaseEncoder) or isinstance(obj, _BaseEncoder)


def isa_encoder(est):
    return isinstance(est, _BaseEncoder)


class BaseReplacer(_BaseEncoder):
    def __init__(self, categories='auto', dtype=np.float64, handle_unknown='ignore'):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def _validate_keywords(self):
        if self.handle_unknown not in ('error', 'ignore'):
            msg = ("handle_unknown should be either 'error' or 'ignore', got {0}.".format(self.handle_unknown))
            raise ValueError(msg)

    def _get_feature(self, X, feature_idx):
        if isinstance(X, DataFrame):
            # pandas dataframes
            return X.iloc[:, feature_idx].values
        # numpy arrays, sparse arrays
        return X[:, feature_idx]

    @abstractmethod
    def _fit_to_numbers(self, X, y):
        self.numbers_ = None
        raise NotImplementedError

    def fit(self, X, y=None, **fit_params):
        self._validate_keywords()
        self._fit(X, handle_unknown=self.handle_unknown)
        self._fit_to_numbers(X, y)
        return self

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        input_features : list of str of shape (n_features,)
            String names for input features if available. By default, "x0", "x1", ... "xn_features" is used.

        Returns
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """
        check_is_fitted(self)
        cats = self.categories
        if input_features is None:
            input_features = ['x%d' % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError("input_features should have length equal to number of features ({}), got {}".format(len(self.categories_), len(input_features)))
        feature_names = input_features
        return np.asarray(feature_names, dtype=object)
    
    def transform(self, X):
        check_is_fitted(self, "numbers_")
        X_int, _ = self._transform(X, handle_unknown=self.handle_unknown)
        numbers = self.numbers_

        n_samples, n_features = X.shape
        Xt = np.empty((n_samples, n_features), dtype=self.dtype)
        for i in range(n_features):
            numbers[i] = np.array(numbers[i])
            Xt[:, i] = numbers[i][X_int[:, i]]
        
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            Xt = DataFrame(data=Xt, columns=columns, index=index)
            if isinstance(self.dtype, dict):
                Xt = Xt.astype(self.dtype)
        return Xt
    
    def inverse_transform(self, X):
        check_is_fitted(self, "numbers_")
        X_columns, n_samples, n_features_ = self._check_X(X)

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} columns, got {1}.")
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        
        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        for i, x, cats, numbers in zip(range(n_features), X_columns, self.categories_, self.numbers_):
            for j, num in enumerate(numbers):
                X[np.equal(x, num)] = j
            labels = x.astype('int64', copy=False)
            X_tr[:, i] = cats[labels]
        
        if isinstance(X, DataFrame):
            columns = x.columns
            index = x.index
            X_tr = DataFrame(data=X_tr, columns=columns, index=index)
        
        return X_tr


class BaseExpander(_BaseEncoder):
    def __init__(self, categories='auto', dtype=np.float64, handle_unknown='ignore', n_jobs=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs
    
    def _validate_keywords(self):
        if self.handle_unknown not in('error', 'ignore'):
            msg = ("handle_unknown should be either 'error' or 'ignore', got {}.".format(self.handle_unknown))
            raise ValueError(msg)
    
    def _get_feature(self, X, feature_idx):
        if isinstance(X, DataFrame):
            # pandas dataframes
            return X.iloc[:, feature_idx].values
        # numpy arrays, sparse arrays
        return X[:, feature_idx]
    
    @abstractmethod
    def _fit_to_numbers(self):
        self.numbers_ = None
        raise NotImplementedError
        # self.numbers_ = Parallel(self.n_jobs)(delayed(fit_to_map)(len(cats)) for cats in self.categories_)
    
    def fit(self, X, y=None, **fit_params):
        self._validate_keywords()
        self._fit(X, handle_unknown=self.handle_unknown)
        self._fit_to_numbers()
        return self

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        -----------
        input_features : list of str of shape (n_features,)
            String names for input features if available. By default, "X0", "x1", ... "xn_features" is used.
        
        Returns
        -----------
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """
        check_is_fitted(self)
        cats = self.categories_
        numbers = self.numbers_

        if input_features is None:
            input_features = ['x%d' % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError("input_features should have length equal to number of features ({}), got {}".format(len(self.categories_), len(input_features)))
        feature_names = []
        for i, input_feature in enumerate(input_features):
            feature_names.extend([f"{input_feature}_{j}" for j in range(numbers[i].shape[1])])
        return np.asarray(feature_names, dtype=object)

    # TODO: accelerate this code
    def transform(self, X):
        check_is_fitted(self, "numbers_")
        n_samples, n_features = X.shape
        X_int, _ = self._transform(X, handle_unknown=self.handle_unknown)
        map_dic = []
        n_encoder_features = 0
        for i in range(len(self.categories_)):
            d_list = []
            number = self.numbers_[i]
            for j in range(number.shape[1]):
                d = dict(zip(range(len(self.categories_[i])), number[:, j]))
                d_list.append(d)
                n_encoder_features += 1
            map_dic.append(d_list)
        Xt = np.empty((n_samples, n_encoder_features), dtype=self.dtype)
        cur = 0
        for i, d_list in enumerate(map_dic):
            for j, dic in enumerate(d_list):
                Xt[:, cur] = np.vectorize(dic.get)(np.array(X_int[:, i]))
                cur += 1

        if isinstance(X, DataFrame):
            columns = self.get_feature_names(X.columns)
            index = X.index
            return DataFrame(data=Xt, columns=columns, index=index)
        
        # for i in range (n_features):
        #     Xt[:, i] = numbers[i][X_int[:, i]]

        return Xt


def is_replacer(obj):
    return issubclass(obj, BaseReplacer) or isinstance(obj, BaseReplacer)


def is_expander(obj):
    return issubclass(obj, BaseExpander) or isinstance(obj, BaseExpander)


def isa_replacer(est):
    return isinstance(est, BaseReplacer)


def isa_expander(est):
    return isinstance(est, BaseExpander)
