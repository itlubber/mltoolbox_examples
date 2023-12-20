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
    >>> from orca_ml.rule_extraction import Rule, SelectUniqueRule
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


