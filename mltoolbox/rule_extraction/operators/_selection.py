import numbers
from abc import abstractmethod
from warnings import warn
import numpy as np
from pandas import DataFrame
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.feature_selection._univariate_selection import _clean_nans
from sklearn. metrics import check_scoring
from sklearn.metrics._classification import recall_score, precision_score
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args
from .._utils import check_rules
from ..metrics._scorer import support_score
from ...base import BaseEstimator, clone
from ...feature_selection._base import SelectorMixin
from ...metrics._lift_score import lift_score


class RuleSelectorMixin(SelectorMixin):
    @abstractmethod
    def _get_support_mask(self):
        """Get the boolean mask indicating which rules are selected

        Returns
        ----------
        support : boolean array of shape [# input rules]
            An element is True iff its corresponding rule is selected for retention.
        """
        pass
    
    def transform(self, R):
        """Reduce R to the selected rules.

        Parameters
        -----------
        R: list of Rule, shape [n_rules, ]
            The input rules.

        Returns
        -----------
        R_r : list of Rule, shape [n_selected_rules, ]
            The selected rules.
        """
        R = check_rules(R)
        mask = self._get_support_mask()
        if not mask.any():
            warn("No rules were selected: either the rule is too noisy or the selection test too strict.", UserWarning)
            return []
        if len(mask) != len(R):
            raise ValueError("R has a different shape than during fitting.")
        return [r for r, selected in zip(R, mask) if selected]

    # TODO: implement me.
    def inverse_transform(self, R):
        """Reverse the transformation operation.

        Parameters
        -----------
        R: list of Rule, shape [n_selected_rules, ]
            The input rules.

        Returns
        -----------
        R_r : list of Rule, shape [n_rules, ]
            R with dummy rule inserted where rules would have been removed by :meth:transform.
        """
        pass


class SelectUniqueRule(BaseEstimator, RuleSelectorMixin):
    """Select unique rule.

    Examples
    -----------
    >>> from mltoolbox.rule_extraction import Rule, SelectUniqueRule
    >>> R = [Rule("f1 > 1"), Rule("f2 < 2"), Rule("f1 > 1")]
    >>> sel = SelectUniqueRule().fit(R)
    >>> sel.transform(R)
    [Rule('f1 > 1'), Rule('f2 < 2')]
    """
    def __init__(self):
        pass

    def fit(self, R, y=None, **fit_params):
        rules = check_rules(R, estimator=self)
        n_rules = len(rules)
        mask = np.full(n_rules, True, dtype=bool)
        if n_rules >= 1:
            for l in range(n_rules):
                if not mask[l]:
                    continue
                for r in range(l + 1, n_rules):
                    if not mask[r]:
                        continue
                    if rules[l] == rules[r]:
                        mask[r] = False
        self.support_mask_ = mask
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_


class BaseRankedRuleSelector(BaseEstimator, RuleSelectorMixin):
    def __init__(self, top_k=None):
        self.top_k = top_k

    @abstractmethod
    def _get_support_mask(self):
        """Get the boolean mask indicating which rules are selected.
        
        Returns
        ----------
        support : boolean array of shape [# input rules]
            An element is True iff its corresponding rule is selected for retention.
        """
        raise NotImplementedError

    def transform(self, R):
        R = check_rules(R)
        mask = self._get_support_mask()
        if not mask.any():
            warn("No rules were selected: either the rule is too noisy or the selection test too strict.", UserWarning)
            return []
        if len(mask) != len(R):
            raise ValueError("R has a different shape than during fitting.")
        # hint: the different part for ranked selector!
        indices = np.argsort(-self.scores_, kind='mergesort')
        # scores[indices] is monotonic decreasing
        if self.top_k is None:
            # only sort the rules
            Rt = [R[idx] for idx in indices]
        else:
            if self.top_k < 0:
                top_k = len(R) - self.top_k
            else:
                top_k = self.top_k
            Rt = [R[idx] for i, idx in enumerate(indices) if i < top_k]
        return Rt


class BaseRuleFilter(BaseEstimator, RuleSelectorMixin):
    def __init__(self, *, score_func=None):
        self.score_func = score_func

    def _check_y(self, y):
        if y is None and self._get_tags()['requires_y']:
            raise ValueError("Rule filter requires y, but not supplied.")
        
        # from sklearn.preprocessing import LabeLEncoder
        # le = LabeLEncoder()
        # у = le.fit_transform(y)
        # if len(le.classes_) != 2:
        #     raise ValueError("Only support binary Label for rule filtering!")
        return y

    def fit(self, R, y=None, **fit_params):
        R= check_rules(R)
        y = self._check_y(y)
        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) was passed." % (self.score_func, type(self.score_func)))
        self._check_params(R, y)
        scores = [self.score_func(y, r.result()) for r in R]
        self.scores_ = np.asarray(scores)
        return self
    
    @abstractmethod
    def _get_support_mask(self):
        pass

    def _check_params(self, R, y):
        pass

    def _more_tags(self):
        return {'requires_y': True}


class SelectRulePercentile(BaseRuleFilter):
    """Select rules according to a percentile of the highest scores.

    Read more in the :ref:'User Guide <univariate_feature_selection>.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays R and y, and returning a single array with scores.
    percentile : int, optional, default=10
        Percent of rules to keep.

    Attributes
    ----------
    scores_ : array-like of shape (n_rules,)
        Scores of rules.
    """
    def __init__(self, score_func=lift_score, *, percentile=10):
        super(SelectRulePercentile, self).__init__(score_func=score_func)
        self.percentile = percentile

    def _check_params(self, R, y):
        if not 0 <= self.percentile <= 100:
            raise ValueError(f"percentile should be >=0, <=100; got {repr(self.percentile)}")

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        # Cater for NaNs
        if self.percentile == 100:
            return np.ones(len(self.scores_), dtype=np.bool)
        elif self.percentile == 0:
            return np.zeros(len(self.scores_), dtype=np.bool)
        
        scores = _clean_nans(self.scores_)
        threshold = np.percentile(scores, 100 - self.percentile)
        mask = scores > threshold
        ties = np.where(scores == threshold)[0]
        if len(ties):
            max_feats = int(len(scores) * self.percentile / 100)
            kept_ties = ties[:max_feats - mask.sum()]
            mask[kept_ties] = True
        return mask


class SelectRuleKBest(BaseRuleFilter):
    """Select rules according to the k highest scores

    Read more in the :ref: 'User Guide <univariate_feature_selection>.

    Parameters
    ----------
    score_func : callable
        Function taking tuo arrays R and y, and returning a single array with scores.
    k : int or "all", optional, default=10
        Number of top rules to select
        The "all" option bypasses selection, for use in a parameter search.

    Attributes
    ----------
    scores_ : array-like of shape (n_rules,)
        Scores of rules.
    """
    def __init__(self, score_func=lift_score, *, k=10):
        super(SelectRuleKBest, self). __init__(score_func=score_func)
        self.k = k
        
    def _check_params(self, R, y):
        if not (self.k == "all" or 0 <= self.k <= len(R)):
            raise ValueError(f"k should be >=0, <- n_rules - {len(R)}; got {repr(self.k)}. Use k = 'all' to return all rules.")
    
    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        if self.k == 'all':
            return np.ones(self.scores_.shape, dtype=bool)
        elif self.k == 0:
            return np.zeros(self.scores_.shape, dtype=bool)
        else:
            scores = _clean_nans(self.scores_)
            mask = np.zeros(scores.shape, dtype=bool)

            # Request a stable sort. Mergesort takes more memory (~40MB per megafeature on x86-64)
            mask[np.argsort(scores, kind="mergesort")[-self.k:]] = 1
            return mask


class ThresholdRuleFilter(BaseRuleFilter):
    def __init__(self, *, score_func=None, threshold=None):
        super(ThresholdRuleFilter, self).__init__(score_func=score_func)
        self.threshold = threshold

    @property
    def threshold_(self):
        check_is_fitted(self, "scores_")
        return _calculate_threshold(self, self.scores_, self.threshold)
    
    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self.threshold)
        return self.scores_ >= threshold


class RulePrecisionSelector(ThresholdRuleFilter):
    def __init__(self, *, threshold=0.8):
        super(RulePrecisionSelector, self).__init__(score_func=precision_score, threshold=threshold)


class RuleRecallSelector(ThresholdRuleFilter):
    def __init__(self, *, threshold=0.8):
        super(RuleRecallSelector, self).__init__(score_func=recall_score, threshold=threshold)


class RuleSupportSelector(ThresholdRuleFilter):
    def __init__(self,*, threshold=0.01):
        super(RuleSupportSelector, self).__init__(score_func=support_score, threshold=threshold)


class RuleliftSelector(ThresholdRuleFilter):
    def __init__(self, *, threshold=3.0):
        super(RuleliftSelector, self).__init__(score_func=lift_score, threshold=threshold)


class RuleCorrSelector(BaseEstimator, RuleSelectorMixin):
    def __init__(self, *, threshold=0.8, method="pearson"):
        self.threshold = threshold
        self.method = method

    def fit(self, R, y=None, **fit_params):
        R = check_rules(R)
        n_rules = len(R)
        X = np.column_stack(tuple(r.result() for r in R))
        X = DataFrame(X)
        corr_matrix = X.corr(method=self.method).values
        self.n_rules_in_ = n_rules
        self.corr_matrix_ = corr_matrix
        return self
    
    def _get_support_mask(self):
        check_is_fitted(self, ["corr_matrix_", "n_rules_in_"])
        threshold = _calculate_threshold(self, self.corr_matrix_[np.triu_indices(self.n_rules_in_, k=1)], self.threshold)
        n_rules = self.n_rules_in_
        corr_matrix = self.corr_matrix_
        mask = np.full(n_rules, True, dtype=np.bool)
        for i in range(n_rules):
            if not mask[i]:
                continue
            for j in range(i + 1, n_rules):
                if not mask[j]:
                    continue
                if abs(corr_matrix[i, j]) < threshold:
                    continue
                mask[j] = False
        return mask
    
    @property
    def threshold_(self):
        check_is_fitted(self, ["corr_matrix_", "n_rules_in_"])
        return _calculate_threshold(self, self.corr_matrix_[np.triu_indices(self.n_rules_in_, k=1)], self.threshold)


class RulePearsonSelector(BaseEstimator, RuleSelectorMixin):
    def __init__(self, *, threshold=0.8):
        self.threshold = threshold

    def fit(self, R, y=None, **fit_params):
        R= check_rules(R)
        n_rules = len(R)
        threshold = self.threshold
        mask = np.full(n_rules, True, dtype=np.bool)
        for i in range(n_rules):
            if not mask[i]:
                continue
            for j in range(i + 1, n_rules):
                if not mask[j]:
                    continue
                ri, rj = R[i], R[j]
                if abs(pearsonr(ri.result(), rj.result())) < threshold:
                    continue
                mask[j] = False
        self.support_mask_ = mask
        self.n_rules_in_= n_rules
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_


def pearsonr(x, y):
    """Calculate a Pearson correlation coefficient.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
    y : array-like of shape (n_samples,)

    Returns
    ----------
    r: float
        Pearson's correlation coefficient

    Examples
    ---------
    >>> import numpy as np
    >>> a = np.array([0,0,0,1,1,1,1])
    >>> b = np.arange(7)
    >>> pearsonr(a, b)
    0.8660254037844386
    """
    mx, my = x.mean(), y.mean()
    xm, ym = x- mx, y - my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    r= r_num / r_den
    r = max(min(r, 1.0), -1.0)
    return r


class RuleIterativeSelector(BaseEstimator, RuleSelectorMixin):
    def __init__(self, *, metrics='support', min_delta_score=0.001, method='delete'):
        self.metrics = metrics
        self.min_delta_score = min_delta_score
        self.method = method
    def fit(self, R, y=None, **fit_params):
        R= check_rules(R)
        n_rules = len(R)
        if self.metrics not in ['support', 'lift', 'recall']:
            raise TypeError("The metrics function should be 'support','lift','recall', %s was passed." % self.metrics)
        if self.method == 'delete':
            x = [r.result() for r in R]
            replacer = np.full(len(x[0]), False, dtype=np.bool)
            y_pred = np.any(X, axis=0)
            lift = lift_score(y_pred=y_pred, y_true=y)
            support = support_score(y_pred=y_pred, y_true=y)
            recall = recall_score(y_pred=y_pred, y_true=y)
            mask = np.full(n_rules, True, dtype=bool)
            for i in range(n_rules):
                mask[i] = False
                tmp = X[i]
                X[i] = replacer
                y_pred = np.any(X, axis=0)
                if self.metrics == 'support':
                    tmp_support = support_score(y_pred=y_pred, y_true=y)
                    support_delta = support - tmp_support
                    if support_delta > self.min_delta_score:
                        mask[i] = True
                        X[i] = tmp
                    else:
                        support = tmp_support
                elif self.metrics == 'lift':
                    tmp_lift = lift_score(y_pred=y_pred, y_true=y)
                    lift_delta = lift - tmp_lift
                    if lift_delta > self.min_delta_score:
                        mask[i] = True
                        X[i] = tmp
                    else:
                        lift = tmp_lift
                elif self.metrics == 'recall':
                    tmp_recall = recall_score(y_pred=y_pred, y_true=y)
                    recall_delta = recall - tmp_recall
                    if recall_delta > self.min_delta_score:
                        mask[i] = True
                        X[i] = tmp
                    else:
                        recall = tmp_recall
            self.support_mask_ = mask
        elif self.method == 'add':
            lift = 0
            support = 0
            recall = 0
            mask = np.full(n_rules, False, dtype=bool)
            X=[]
            for i in range(n_rules):
                mask[i] = True
                X. append(R[i].result())
                y_pred = np.any(X, axis=0)
                if self.metrics == 'support':
                    tmp_support = support_score(y_pred=y_pred, y_true=y)
                    support_delta = tmp_support - support
                    if support_delta < self.min_delta_score:
                        mask[i] = False
                        X.pop()
                    else:
                        support = tmp_support
                elif self.metrics == 'lift':
                    tmp_lift = lift_score(y_pred=y_pred, y_true=y)
                    lift_delta = lift - tmp_lift
                    if lift_delta < self.min_delta_score:
                        mask[i] = False
                        X.pop()
                    else:
                        lift = tmp_lift
                elif self.metrics == 'recall':
                    tmp_recall = recall_score(y_pred=y_pred, y_true=y)
                    recall_delta = recall - tmp_recall
                    if recall_delta < self.min_delta_score:
                        mask[i] = False
                        X.pop()
                    else:
                        recall = tmp_recall
            self.support_mask_ = mask

        self.n_rules_in_ = n_rules
        return self

    def _get_support_mask(self):
        check_is_fitted(self, ["support_mask_", "n_rules_in_"])
        return self.support_mask_


class RecursiveRuleElimination(BaseEstimator, RuleSelectorMixin):
    """Rule ranking with recursive rule elimination.

    Given an RuleClassifier that assigns weights to rules (e.g. the
    rule metrics), the goal of recursive rule elimination (RRE) is
    to select rules by recursively considering smaller and smaller
    sets of rules. First, the estimator is trained on the initial set
    of rules and the importance of each rule is obtained either through
    any callable rule scorer .
    Then, the least important rules are pruned from current set of rules.
    That procedure is recursively repeated on the pruned set until the
    desired number of rules to select if eventually reached.

    Read more in the :ref:'User Guide <rre>.

    Parameters
    -----------
    estimator :  RuleClassifier instance
        A rule classifier.
    
    n_rules_to_select : int or float, default=None
        The number of rules to select. If 'None , half of the rules are selected. If integer, the parameter is the absolute number of rules
        to select. If float between 0 and 1, it is the fraction of rules to select.
    
    step : int or float, default=1
        If greater than or equal to 1, then step corresponds to the (integer) number of rules to remove at each iteration.
        If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of rules to remove at each iteration.
    """
    @_deprecate_positional_args
    def __init__(self, estimator, *, scoring=None, n_rules_to_select=None, step=1) :
        self.estimator = estimator
        self.scoring = scoring
        self.n_rules_to_select = n_rules_to_select
        self.step = step
    
    def fit(self, R,y):
        """Fit the RRE model and then the underlying estimator on the selected rules.

        Parameters
        -----------
        R: list
            The training input rules.
        y : array-like of shape (n_samples,)
            The target values.
        
        Returns
        -----------
        self
        """
        return self._fit(R, y)
    
    def _fit(self, R,y):
        scorer = check_scoring(self.estimator, self.scoring)
        R= check_rules(R)
        error_msg =(f"n_rules_to_select must be either None, a positive integer representing the absolute number of rules or a float in (0.0, 1.0] representing a percentage of rules to select. Got {self.n_rules_to_select}")
        
        # Initialization
        n_rules = len(R)
        if self.n_rules_to_select is None:
            n_rules_to_select = n_rules // 2
        elif self.n_rules_to_select <0:
            raise ValueError(error_msg)
        elif isinstance(self.n_rules_to_select, numbers. Integral): # int
            n_rules_to_select = self.n_rules_to_select
        elif self.n_rules_to_select > 1.0: # float > 1
            raise ValueError(error_msg)
        else:
            n_rules_to_select = int(n_rules * self.n_rules_to_select)
        
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_rules))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")
    
        support_ = np.ones(n_rules, dtype=bool)
        ranking_ = np.ones(n_rules, dtype=int)

        # Elimination
        while np.count_nonzero(support_) > n_rules_to_select:
            # Remaining rules
            rule_indices = np.arange(n_rules)[support_]
            # Rank the remaining rules
            importances = np.empty(rule_indices.shape, dtype=np.float32)
            estimator = clone(self.estimator)
            estimator.fit(R, y)
            base_score = scorer(estimator, R, y)
            for i, idx in enumerate(rule_indices):
                R_remain = [R[j] for j in rule_indices if j != idx]
                estimator = clone(self.estimator)
                estimator.fit(R_remain, y)
                importances[i] = abs(base_score - scorer(estimator, R_remain, y))

            ranks = np.argsort(importances)

            # Eliminate the worse rules
            threshold = min(step, np.count_nonzero(support_) - n_rules_to_select)
            support_[rule_indices[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        rules_indices = np.arange(n_rules)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit([R[i] for i in rules_indices], y)

        self.n_rules_in_ = n_rules
        self.support_ = support_
        self.ranking_ = ranking_
        
        return self

    def _get_support_mask(self):
        check_is_fitted(self, ["support_", "n_rules_in_"])
        return self.support_

    def _more_tags(self):
        return {
            'requires_y': True,
        }


RRE = RecursiveRuleElimination
