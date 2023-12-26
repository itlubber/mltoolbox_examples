from ._rule import Rule


def check_rules(rules, *, ensure_min_rules=1, estimator=None):
    for r in rules:
        if not isinstance(r, Rule):
            raise TypeError(f"Expect a Rule, got {type(r)} instead.")
    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    context = " by %s" % estimator_name if estimator is not None else ""
    if ensure_min_rules > 0:
        n_rules = len(rules)
        if n_rules < ensure_min_rules:
            raise ValueError("Found list with %d rule(s) while a minimum of %d is required%s." % (n_rules, ensure_min_rules, context))
    return rules
