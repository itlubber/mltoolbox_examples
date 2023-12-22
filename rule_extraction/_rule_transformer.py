"""
The :mod:`orca_ml.rule_extraction._rule_transformer` module implements utilities to work with heterogeneous rules and to apply different transformers to different rules.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import _transform_one, _fit_transform_one, _name_estimators
from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.validation import check_is_fitted

from .operators import RuleTransformerMixin
from .operators._rule_cascade_selector import RuleCascadeSelector
from ..base import clone
from ._utils import check_rules
from ..utils._estimator_html_repr import _VisualBlock


_ERR_MSG_IDCOLUMN = ("1D data passed to a transformer that expects 2D data. Try to specify the column selection as a list of one item instead of a scalar.")


def _flatten(ruless):
    """Flatten nested rules

    Parameters
    -----------
    ruless : list of list of Rule.

    Yields
    -----------
    rule : single Rule.

    Examples
    -----------
    >>> ruless = [[Rule("f1 <= 1"), Rule("f1 > 2")], [Rule("f2 > 3.5"), Rule("f2 <= 0.5")]]
    >>> [_ for _ in _flatten(ruless)]
    [Rule("f1 <= 1", Rule("f1 > 2" , Rule("f2 > 3.5", Rule("f2<= 0.5")]
    """
    for rules in ruless:
        if isinstance(rules, list):
            yield from _flatten(rules)
        else:
            yield rules


class RuleTransformer(RuleTransformerMixin, _BaseComposition):
    """Applies transformers to list of Rule

    This estimator allows different rules or rule subsets of the input to be transformed separately and the rules generated by each transformer
    will be concatenated to form a single rule list. This is useful for heterogeneous rules, to combine several rule operations into a single transformer.

    Read more in the :ref:'User Guide <rule_transformer>.

    Examples
    -----------
    >>> import numpy as np
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import Normalizer
    >>> ct = ColumnTransformer([("norm1", Normalizer(norm='l1'), [0, 1]), ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    >>> X = np.array([[0., 1., 2., 2.],
                      [1., 1., 0., 1.]])
    >>> # Normalizer scales each row of x to unit norm. A separate scaling
    >>> # is applied for the two first and two last elements of each
    >>> # row independently.
    >>> ct.fit_transform(X)
    array([[0., 1. , 0.5, 0.5],
           [0.5, 0.5, 0., 1. ]])
    """
    _required_parameters = ['transformers']

    @_deprecate_positional_args
    def __init_(self, transformers, *, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

    @property
    def _transformers(self):
        """Internal list of transformer only containing the name and transformers, dropping the columns. This is for the implementation
        of get_params via BaseComposition._get_params which expects lists of tuples of len 2.
        """
        return [(name, trans) for name, trans, _ in self.transformers]
    
    @_transformers.setter
    def _transformers(self, value):
        self.transformers = [(name, trans, cas) for ((name, trans), (_, _, cas)) in zip(value, self.transformers)]

    def get_params(self, deep=True):
        """Get parameters for this estimator

        Parameters
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.
            
        Returns
        -----------
        params : dict
            Parameter names mapped to their values.
        """
        return self._get_params('_transformers', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator .

        Valid parameter keys can be listed with get_params()

        Returns
        -----------
        self
        """
        self._set_params('_transformers', **kwargs)
        return self
    
    def _iter(self, fitted=False, replace_strings=False):
        """Generate (name, trans, column, weight) tuples.

        If fitted=True, use the fitted transformers, else use the user specified transformers updated with converted column names and potentially appended with transformer for remainder
        """
        if fitted:
            transformers = self.transformers_
        else:
            # interleave the validated column specifiers
            transformers = [(name, trans, cascade) for (name, trans, _), cascade in zip(self.transformers, self._cas)]

        get_weight = (self.transformer_weights or {}).get

        for name, trans, cascade in transformers:
            if _is_empty_cascade_selection(cascade):
                continue
            yield (name, trans, cascade, get_weight(name))

    def _validate_transformers(self):
        if not self.transformers:
            return

        names, transformers, _ = zip(*self.transformers)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ('drop', 'passthrough'):
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and transform, or can be 'drop' or 'passthrough' specifiers. '%s' (type %s) doesn't." % (t,type(t)))

    def _validate_selector_callables(self, R):
        """Converts callable rule selector specifications."""
        cas = []
        for _, _, cascade in self.transformers:
            if callable(cascade):
                cascade = cascade(R)
            cas.append(cascade)
        self._cas = cas

    @property
    def named_transformers_(self):
        """Access the fitted transformer by name.

        Read-only attribute to access any transformer by given name Keys are transformer names and values are the fitted transformer objects.
        """
        # Use Bunch object to improve autocomplete
        return Bunch(**{name: trans for name, trans, _ in self.transformers_})
    
    def _update_fitted_transformers(self, transformers):
        # transformers are fitted; excludes 'drop' cases
        fitted_transformers = iter(transformers)
        transformers_ = []

        for name, old, cascade, _ in self._iter():
            if _is_empty_cascade_selection(cascade):
                trans = old
            else:
                trans = next(fitted_transformers)
            transformers_.append((name, trans, cascade))

        # sanity check that transformers is exhaustea
        assert not list(fitted_transformers)
        self.transformers_ = transformers_

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(%d of %d) Processing %s' % (idx, total, name)

    def _fit_transform(self, R, y, func, fitted=False):
        """Private function to fit and/or transform on demand.

        Return value (transformers and/or transformed X data) depends on the passed function.
        fitted=True ensures the fitted transformers are used.
        """
        transformers = list(self._iter(fitted=fitted, replace_strings=True))
        try:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(
                    transformer=clone(trans) if not fitted else trans,
                    X=RuleCascadeSelector(cascade=cascade).select(R),
                    y=y,
                    weight=weight,
                    message_clsname="RuleTransformer",
                    message=self._log_message(name, idx, len(transformers)))
                for idx, (name, trans, cascade, weight) in enumerate(self._iter(fitted=fitted, replace_strings=True), 1))
        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_IDCOLUMN)
            else:
                raise

    def fit(self, R, y=None, **fit_params):
        self.fit_transform(R, y=y)
        return self

    def fit_transform(self, R, y=None, **fit_params):
        """Fit all transformers, transform the rule and concatenate results.

        Parameters
        -----------
        R: list of Rule, length = n_rules
            Input rules, of which specified subsets are used to fit the transformers.

        y : array-like of shape (n_samples,), default=None
            Targets for supervised learning.

        fit_params : dict
            Exist for pipeline compatibility.

        Returns
        -----------
        Rt: list of Rule, length = n_rules_out
            hstack of results of transformers. n_rules_out is the number of all transformed rules.
        """
        R = check_rules(R, estimator=self)
        n = len(R)
        self.n_rules_in_ = n

        # set n_features_in_ attribute
        self._validate_transformers()
        self._validate_selector_callables(R)

        result = self._fit_transform(R, y, _fit_transform_one)

        if not result:
            self._update_fitted_transformers([])
            # Al transformers are None
            return []
        
        Rs, transformers = zip(*result)

        self._update_fitted_transformers(transformers)

        return self._hstack(Rs)

    def transform(self, R):
        """Transform R separately by each transformer, concatenate results.

        Parameters
        -----------
        R : list of Rule, length = n_rules
            The data to be transformed by subset.

        Returns
        -----------
        R_t : list of Rule, length = n_rules_out
            hstack of results of transformers. n_rules_out is the number of all transformed rules.
        """
        check_is_fitted(self)
        R= check_rules(R, estimator=self)

        # TODO: also call_check_n_features(reset=False) in 0.24

        Rs = self._fit_transform(R, None, _transform_one, fitted=True)

        if not Rs:
            # All transformers are None
            return []
        
        return self._hstack(Rs)

    def _hstack(self, Rs):
        """Stacks Rs horizontally.

        This allows subclasses to control the stacking behavior, while reusing everything else from RuleTransformer.

        Parameters
        -----------
        Rs : list of list of Rule

        Returns
        -----------
        R : list of Rule, length = n_rules_out
        """
        return [_ for _ in _flatten(Rs)]
    
    def _sk_visual_block_(self):
        names, transformers, name_details = zip(*self.transformers)
        return _VisualBlock('parallel', transformers,
    names=names, name_details=name_details)


def _is_empty_cascade_selection(cascade):
    """Return True if the column selection is empty (empty list or all-False boolean array)."""
    if isinstance(cascade, int) or isinstance(cascade, float):
        if cascade > 0:
            return False
    return True


def _get_transformer_list(estimators):
    """Construct (name, trans, column) tuples from list"""
    transformers, cascade = zip(*estimators)
    names, _ = zip(*_name_estimators(transformers))

    transformer_list = list(zip(names, transformers, cascade))
    return transformer_list


def make_rule_transformer(*transformers, **kwargs):
    """
    Examples
    -----------
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> make_column_transformer((Standardscaler(), ['numerical_column']), (OneHotEncoder(), ["categorical_column"]))
    ColumnTransformer(transformers=[('standardscaler', Standardscaler(...), ['numerical_column']), 
                                    ('onehotencoder', OneHotEncoder(...), ['categorical_column'])])
    """
    # transformer_weights keyword is not passed through because the user would need to know the automatically generated names of the transformers
    n_jobs = kwargs.pop('n_jobs', None)
    remainder = kwargs.pop('remainder', 'drop')
    sparse_threshold = kwargs.pop('sparse_threshold', 0.3)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'.format(list(kwargs. keys())[0]))
    
    transformer_list = _get_transformer_list(transformers)
    return RuleTransformer(transformer_list, n_jobs=n_jobs, remainder=remainder, sparse_threshold=sparse_threshold, verbose=verbose)


class make_rule_selector:
    """Create a callable to select columns to be used with :class: ColumnTransformer.
    
    :func: `make_column_selector` can select columns based on datatype or the
    columns name with a regex. when using multiple selection criteria, **all**
    criteria must match for a column to be selected.
    
    Parameters
    -----------
    pattern : str, default=None
        Name of columns containing this regex pattern will be included. If None, column selection will not be selected based on pattern.
    
    Returns
    -----------
    selector : callable
        Callable for column selection to be used by a :class: ColumnTransformer'.
    
    See also
    -----------
    sklearn.compose.ColumnTransformer: Class that allows combining the outputs of multiple transformer objects used on column subsets of the data into a single feature space.
    
    Examples
    -----------
    >>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> from sklearn.compose import make_column_selector
    >>> import pandas as pd # doctest: +SKIP
    >>> X = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'], 'rating': [5, 3, 4, 5]}) # doctest: +SKIP
    >>> ct = make_column_transformer((StandardScaler(), make_column_selector(dtype_include=np.number)), # rating
                                     (OneHotEncoder(), make_column_selector(dtype_include=object))) # city
    >>> ct.fit_transform(X) # doctest: +SKIP
    """
    @_deprecate_positional_args
    def __init__(self, pattern=None, *, cascade=None):
        self.pattern = pattern
        self.cascade = cascade
    
    def __call__(self):
        return self.cascade
