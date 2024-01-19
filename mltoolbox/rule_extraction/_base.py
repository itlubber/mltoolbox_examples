from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from ._rule import Rule
# from ._memory_effident_rule import Rule
from ..base import BaseEstimator, FrameTransformerMixin


def _flatten(ruless):
    """Flatten nested rules.

    Parameters
    -----------
    ruless : list of rules.

    Returns
    -----------
    rules : list of rule

    Examples
    -----------
    >>> ruless = [[Rule("f1 <= 1"), Rule("f1 > 2")], [Rule("f2 > 3.5"), Rule("f2 <= 0.5")]]
    >>> [_ for _ in _flatten(ruless)]
    [Rule("f1 <= 1"), Rule("f1 > 2"), Rule("f2 > 3.5"), Rule("f2 <= 0.5")]
    """
    for rules in ruless:
        if isinstance(rules, list):
            yield from _flatten(rules)
        else:
            yield rules


class BaseRuleExtractor(BaseEstimator, FrameTransformerMixin):
    _closed = "left"

    @_deprecate_positional_args
    def __init__(self, *, n_bins=5):
        self.n_bins = n_bins

    def _check_y(self, y):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        return y
    
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
        raise AttributeError("Can not delete this attribute!")
    
    def transform(self, X):
        X = self._ensure_dataframe(X)
        rules = self._get_extracted_rules()
        self._trigger_rule_prediction(rules, X)
        return rules
    
    def _get_extracted_rules(self):
        """Get extracted rules.

        Returns
        ----------
        rules : List[Rule]
            List of extracted rules.
        """
        check_is_fitted(self, "rules_")
        return [r for r in _flatten(self.rules_)]
    
    def _trigger_rule_prediction(self, rules, X):
        """Trigger rule prediction for further usage.

        Parameters
        ----------
        rules
        X
        """
        for rule in rules:
            if not isinstance(rule, Rule):
                raise ValueError("Input should be `Rule`.")
            rule.predict(X)
