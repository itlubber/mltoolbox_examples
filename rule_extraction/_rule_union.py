import warnings
from joblib import Parallel, delayed
from sklearn.pipeline import _fit_one, _transform_one, _fit_transform_one, _name_estimators, _VisualBlock
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import _deprecate_positional_args

from .operators._base import RuleTransformerMixin


def _flatten(ruless):
    """Flatten nested rules.

    Parameters
    ------------
    ruless : list of list of Rule

    Yields
    ------------
    rule : single Rule.

    Examples
    ------------
    >>> ruless = [[Rule("f1 <= 1"), Rule("f1 > 2")], [Rule("f2 > 3.5"), Rule("f2 <= 0.5")]]
    >>> [_ for _ in _flatten(ruless)]
    [Rule("f1 <= 1"), Rule("f1 > 2"), Rule("f2 >3.5"), Rule("f2 <= 0.5")'
    """
    for rules in ruless:
        if isinstance(rules, list):
            yield from _flatten(rules)
        else:
            yield rules


class RuleUnion(RuleTransformerMixin, _BaseComposition):
    _required_parameters = ["transformer_list"]

    @_deprecate_positional_args
    def __init__(self, transformer_list, *, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self._validate_transformers()

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ------------
        deep : bool, default-True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        ------------
        params : mapping of string to any
            Parameter names mapped to their values
        """
        return self._get_params('transformer_list', deep=deep)
    
    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with get_params().

        Returns
        ------------
        self
        """
        self._set_params('transformer_list', **kwargs)
        return self
    
    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            # TODO: Remove in 0.24 when None is removed
            if t is None:
                warnings.warn("Using None as a transformer is deprecated in version 0.22 and will be removed in version 0.24. Please use 'drop' instead.", FutureWarning)
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and transform. '%s' (type %s) doesn't" % (t, type(t)))
    
    def _iter(self):
        """Generate (name, trans, weight) tuples excluding None and 'drop' transformers."""
        get_weight = (self.transformer_weights or {}).get
        return ((name, trans, get_weight(name)) for name, trans in self.transformer_list if trans is not None and trans != 'drop')

    def fit(self, R, y=None, **fit_params):
        """Fit all rule transformers using R.

        Parameters
        ------------
        R: list, each element is a Rule
            Input rules, used to fit transformers.
        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        Returns
        ------------
        self : RuleUnion
            This estimator
        """
        transformers = self._parallel_func(R, y, fit_params, _fit_one)
        if not transformers:
            # All transformers are None
            return self
        
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, R, y=None, **fit_params):
        results = self._parallel_func(R, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return []
        
        Rs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        return [_ for _ in _flatten(Rs)]

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(step %d of %d) Processing %s' % (idx, total, name)
    
    def _parallel_func(self, R, y, fit_params, func):
        """Runs func in parallel on X and y"""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = list(self._iter())

        return Parallel(n_jobs=self.n_jobs)(delayed(func)(
            transformer, R, y, weight, message_clsname='RuleUnion', message=self._log_message(name, idx, len(transformers)), **fit_params) 
            for idx, (name, transformer, weight) in enumerate(transformers, 1))
    
    def transform(self, R):
        """Transform R separately by each rule transformer, concatenate results

        Parameters
        -----------
        R: iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -----------
        R_t : list
            hstack results of all rule transformers.
        """
        Rs = Parallel(n_jobs=self.n_jobs)(delayed(_transform_one)(trans, R, None, weight) for name, trans, weight in self._iter())
        if not Rs:
            # All transformer are None
            return []
        return [_ for _ in _flatten(Rs)]

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [(name, old if old is None or old == 'drop' else next(transformers)) for name, old in self.transformer_list]

    @property
    def n_rules_in_(self):
        # X is passed to all transformers so we just delegate to the first one
        return self.transformer_list[0][1].n_rules_in_

    def _sk_visual_block_(self):
        names, transformers = zip(*self.transformer_list)
        return _VisualBlock('parallel', transformers, names=names)


def make_rule_union(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', None)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        # We do not currently support transformer_weights as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0]))
    return RuleUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
