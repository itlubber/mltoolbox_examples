from itertools import islice
import numpy as np
from sklearn.base import TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.pipeline import _name_estimators
from sklearn.utils import Bunch
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.metaestimators import _BaseComposition


class RandomChoice(_BaseComposition):
    """Random choice of estimators with choice probability.

    Examples
    -----------
    >>> from orca_ml.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    >>> from orca_ml.pipeline import RandomChoice
    >>> from orca_ml.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> rc - RandomChoice(t('s1', MinMaxScaler()), ('s2', RobustScaler()), ('s3', StandardScaler)])
    >>> rc.fit(X,y)
    >>> rc.chosen_name_
    """
    _required_parameters = ["candidates"]

    @_deprecate_positional_args
    def __init__(self, candidates, *, replace=True, p=None):
        self.candidates = candidates
        self.replace = replace
        self.p = p
        self._validate_candidates()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobiects that are estimators.
        
        Returns
        -----------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('candidates', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator

        Valid parameter keys can be listed with get_params()

        Returns
        -----------
        self
        """
        self._set_params('candidates', **kwargs)
        return self

    def _validate_candidates(self):
        names, estimators = zip(*self.candidates)

        # validate names
        self._validate_names(names)

        # validate estimators
        if not (all(isinstance(t, TransformerMixin) for t in estimators if t != "drop") or
                all(isinstance(t, ClassifierMixin) for t in estimators if t != "drop") or
                all(isinstance(t, RegressorMixin) for t in estimators if t != "drop")):
            raise TypeError("All estimators should be transformer or classifier or regressor!")
    
    def _iter(self):
        """Generate (idx, (name, trans)) tuples from self.candidates"""
        stop = len(self.candidates)
        for idx, (name, trans) in enumerate(islice(self.candidates, 0, stop)):
            yield idx, name, trans
    
    def _len_(self):
        """Returns the length of the Pipeline"""
        return len(self.candidates)
    
    def __getitem__(self, ind):
        """Returns a sub-random-choice or a single estimator in the random-choice

        Indexing with an integer will return an estimator; using a slice returns another random-choice instance which copies a slice of this
        random-choice. This copy is shallow: modifying (or fitting) estimators in the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in step will not affect a copy.
        """
        if isinstance(ind, slice):
            # if ind.step not in (1, None):
            #     raise ValueError('RandomChoice slicing only supports a step of 1')
            return self.__class__(self.candidates[ind])
        try:
            name, est = self.candidates[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_candidates[ind]
        return est
    
    @property
    def _estimator_type(self):
        final_estimator = self.steps[-1][1]
        if hasattr(final_estimator, "_estimator_type"):
            return final_estimator._estimator_type
        return "transformer"
    
    @property
    def named_candidates(self):
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.candidates))
    
    def _check_fit_params(self, **fit_params):
        fit_params_candidates = {name: {} for name, step in self.candidates if step is not None}
        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError("RandomChoice.fit does not accept the {} parameter. You can pass parameters to specific steps of your "
                                 "RandomChoice using the stepname__parameter format, e.g. RandomChoice.fit(x, y, logisticregression__sample_weight=sample_weight).".format(pname))
            step, param = pname.split('__', 1)
            fit_params_candidates[step][param] = pval
        return fit_params_candidates
    
    def fit(self, X, y=None, **fit_params):
        fit_params_candidates = self._check_fit_params(**fit_params)
        names, estimators = zip(*self.candidates)
        name = np.random.choice(names, replace=self.replace, p=self.p)
        estimator = self.named_candidates[name]
        self.chosen_name_ = name
        self.chosen_estimator_ = estimator.fit(X, y, **fit_params_candidates[name])
        return self

    # @if_delegate_has_method(delegate='choosed_estimator_')
    def transform(self, X):
        return self.chosen_estimator_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).transform(X)

    # @if_delegate_has_method(delegate='choosed_estimator_')
    def predict(self, X):
        return self.chosen_estimator_.predict(X)

    def fit_predict(self, X, y=None, **fit_params):
        return self.fit(X, y=y, **fit_params).predict(X)

    # @if_delegate_has_method(delegate='choosed_estimator_')
    def predict_proba(self, X):
        return self.chosen_estimator_.predict_proba(X)

    def _more_tags(self):
        return {'non_deterministic': True}


def make_random_choice(*candidates, **kwargs):
    replace = kwargs.pop("replace", True)
    p = kwargs.pop("p", None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}".'.format(list(kwargs.keys())[0]))
    return RandomChoice(_name_estimators(candidates), replace=replace, p=p)
