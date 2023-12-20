import itertools
from operator import ior, iand

from sklearn.utils.validation import check_is_fitted
from ._base import RuleTransformerMixin
from .._utils import check_rules
from ...base import BaseEstimator


class RuleIntersection(BaseEstimator, RuleTransformerMixin):
    def __init__(self, depth=None, operators=(ior, iand)):
        self.depth = depth
        self.operators = operators

    def fit(self, R, y=None, **fit_params):
        n_rules = len(R)
        self.n_rules_in_ = n_rules
        depth = min(n_rules, self.depth)
        operators = self.operators
        self.rules_ = []
        if depth > 0:
            append_rule = self.rules_.append
            for rules in itertools.combinations(R, depth):
                for ops in itertools.product(*(operators for _ in range(depth - 1))):
                    append_rule(make_rule(rules, ops))

        return self

    def transform(self, R):
        check_is_fitted(self, 'rules_')
        R = check_rules(R, estimator=self)
        if len(R) != self.n_rules_in_:
            raise ValueError("R has a different shape than during fitting.")
        return self.rules_


def make_rule(rules, ops):
    if len(rules) == 1:
        return rules[0]
    r1, r2, *left_rules = rules
    op, *left_ops = ops
    return make_rule((op(r1, r2), *left_rules), left_ops)
