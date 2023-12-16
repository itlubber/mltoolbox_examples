import warnings
from abc import abstractmethod

import numpy as np
from sklearn.tree._tree import TREE_UNDEFINED
from sklearn.utils import _deprecate_positional_args

from ._base import BaseDiscretizer, BaseShrinkByInflectionDiscretizer
from ._base import _MIN_BIN_WIDTH, _ATOL, _RTOL
from ...base import clone
from ...tree import DecisionTreeClassifierParameterProxy, ExtraTreeClassifierParameterProxy
from ...tree import DecisionTreeRegressorParameterProxy, ExtraTreeRegressorParameterProxy


class BaseDecisionTreeDiscretizer(BaseDiscretizer):
    _closed = "right"

    def fit(self, X, y=None, **fit_params):
        self._make_estimator()
        return super(BaseDecisionTreeDiscretizer, self).fit(X, y=y, **fit_params)
    
    @abstractmethod
    def _make_estimator(self):
        self.estimator = None

    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        col_min, col_max = x.min(), x.max()

        if col_min == col_max:
            warnings.warn("Feature {(} is constant and will be replaced with 0.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            n_bin = 1
            return bin_edge, n_bin
        
        dt = clone(self.estimator)
        dt.max_leaf_nodes = n_bin
        dt.fit(x[ :, None], y)
        thresholds = sorted(s for s in dt.tree_.threshold if s != TREE_UNDEFINED)
        bin_edge = np.r_[col_min, thresholds, col_max]

        # Remove bins whose width are too small (i.e., <= 1e-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
        bin_edge = bin_edge[mask]

        if len(bin_edge) - 1 != n_bin:
            warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
            n_bin = len(bin_edge) - 1
        return bin_edge, n_bin
    
    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_y": True,
        }
    

class DecisionTreeClassifierDiscretizer(DecisionTreeClassifierParameterProxy, BaseDecisionTreeDiscretizer):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None, criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight=None, ccp_alpha=0.0):
        BaseDecisionTreeDiscretizer.__init__(self, n_bins=n_bins, n_jobs=n_jobs)
        DecisionTreeClassifierParameterProxy.__init__(
            self,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha
        )


class DecisionTreeRegressorDiscretizer(DecisionTreeRegressorParameterProxy, BaseDecisionTreeDiscretizer):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None,
                criterion="mse",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                ccp_alpha=0.0):
        BaseDecisionTreeDiscretizer._init_(self, n_bins=n_bins, n_jobs=n_jobs)
        DecisionTreeRegressorParameterProxy.__init__(
                                                    self,
                                                    criterion=criterion,
                                                    splitter=splitter,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    random_state=random_state,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split,
                                                    ccp_alpha=ccp_alpha,
                                                    )


class ExtraTreeClassifierDiscretizer(ExtraTreeClassifierParameterProxy, BaseDecisionTreeDiscretizer):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None,
                criterion="gini",
                splitter="random",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                class_weight=None,
                ccp_alpha=0.0):
        BaseDecisionTreeDiscretizer.__init__(self, n_bins=n_bins, n_jobs=n_jobs)
        ExtraTreeClassifierParameterProxy.__init__(
                                                    self,
                                                    criterion=criterion,
                                                    splitter=splitter,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    class_weight=class_weight,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split,
                                                    random_state=random_state,
                                                    ccp_alpha=ccp_alpha
                                                )


class ExtraTreeRegressorDiscretizer(ExtraTreeRegressorParameterProxy, BaseDecisionTreeDiscretizer):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None,
                criterion="mse",
                splitter="random",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                random_state=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                max_leaf_nodes=None,
                ccp_alpha=0.0):
        BaseDecisionTreeDiscretizer.__init__(self, n_bins=n_bins, n_jobs=n_jobs)
        ExtraTreeRegressorParameterProxy.__init__(
                                                    self,
                                                    criterion=criterion,
                                                    splitter=splitter,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    random_state=random_state,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    ccp_alpha=ccp_alpha)


class BaseDecisionTreeDiscretizerShrinkByInflection(BaseShrinkByInflectionDiscretizer):
    def fit(self, X, y=None, **fit_params):
        self._make_estimator()
        return super(BaseDecisionTreeDiscretizerShrinkByInflection, self).fit(X, y=y, **fit_params)
    
    @abstractmethod
    def _make_estimator(self):
        self.estimator = None
    
    def _bin_one_column(self, i, n_bin, x, y=None, n_inflection=None):
        col_min, col_max = x.min(), x.max()

        if col_min == col_max:
            warnings.warn("Feature {} is constant and will be replaced with 0.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            n_bin = 1
            return bin_edge, n_bin
        
        if self.closed == 'right':
            right = True
        else:
            right = False
        
        eps = _ATOL + _RTOL * np.abs(x)

        bad_mask, good_mask = np.equal(y, 1), np.not_equal(y, 1)
        bad_tot, good_tot = np.count_nonzero(bad_mask) + 2, np.count_nonzero(good_mask) + 2
        residue = np.log(good_tot / bad_tot)

        dt = clone(self.estimator)

        while True:
            dt.max_leaf_nodes = n_bin
            dt.fit(x[:, None], y)
            thresholds = sorted(s for s in dt.tree_.threshold if s != TREE_UNDEFINED)
            bin_edge = np.r_[col_min, thresholds, col_max]

            # Remove bins whose width are too small (i.e., <= 1e-8)
            mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
            bin_edge = bin_edge[mask]

            if len(bin_edge) - 1 != n_bin:
                warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. consider. decreasing the number of bins.'.format(i))
                n_bin = len(bin_edge) - 1

            if n_bin <= 2:
                break

            # Values which are close to a bin edge are susceptible to numeric instability. Add eps to X so these values are binned correctly
            # with respect to their decimal truncation. See documentation of numpy.isclose for an explanation of rtol and atol.
            xt = np.digitize(x + eps, bin_edge[1:], right=right)
            np.clip(xt, 0, n_bin - 1, out=xt)

            nums = np.zeros(n_bin, dtype=np.float64)
            for j in range(n_bin):
                bin_mask = np.equal(xt, j)
                # ignore unique values, this helps to prevent overfitting on id-like columns.
                if np.count_nonzero(mask) == 1:
                    nums[j] = 0.
                else:
                    bad_num = np.count_nonzero(np.logical_and(bin_mask, bad_mask)) + 1
                    good_num = np.count_nonzero(np.logical_and(bin_mask, good_mask)) + 1
                    nums[j] = np.log(bad_num / good_num) + residue
            
            diffs = np.ediff1d(nums)
            curr_inflection = np.count_nonzero(diffs[1:] * diffs[:-1] < 0)
            if curr_inflection <= n_inflection:
                break
            n_bin -= 1
        return bin_edge, n_bin


class DecisionTreeClassifierDiscretizerShrinkByInflection(DecisionTreeClassifierParameterProxy, BaseDecisionTreeDiscretizerShrinkByInflection):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None, n_inflections=1,
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                class_weight=None,
                ccp_alpha=0.0):
        BaseDecisionTreeDiscretizerShrinkByInflection.__init__(self, n_bins=n_bins, n_jobs=n_jobs, n_inflections=n_inflections)
        DecisionTreeClassifierParameterProxy.__init__(
                                                        self,
                                                        criterion=criterion,
                                                        splitter=splitter,
                                                        max_depth=max_depth,
                                                        min_samples_split=min_samples_split,
                                                        min_samples_leaf=min_samples_leaf,
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                        max_features=max_features,
                                                        max_leaf_nodes=max_leaf_nodes,
                                                        class_weight=class_weight,
                                                        random_state=random_state,
                                                        min_impurity_decrease=min_impurity_decrease,
                                                        min_impurity_split=min_impurity_split,
                                                        ccp_alpha=ccp_alpha)


class DecisionTreeRegressorDiscretizerShrinkByInflection(DecisionTreeRegressorParameterProxy, BaseDecisionTreeDiscretizerShrinkByInflection):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None, n_inflections=1,
                criterion="mse",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                ccp_alpha=0.0):
        BaseDecisionTreeDiscretizerShrinkByInflection.__init__(self, n_bins=n_bins, n_jobs=n_jobs, n_inflections=n_inflections)
        DecisionTreeRegressorParameterProxy.__init__(
                                                        self,
                                                        criterion=criterion,
                                                        splitter=splitter,
                                                        max_depth=max_depth,
                                                        min_samples_split=min_samples_split,
                                                        min_samples_leaf=min_samples_leaf,
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                        max_features=max_features,
                                                        max_leaf_nodes=max_leaf_nodes,
                                                        random_state=random_state,
                                                        min_impurity_decrease=min_impurity_decrease,
                                                        min_impurity_split=min_impurity_split,
                                                        ccp_alpha=ccp_alpha)


class ExtraTreeClassifierDiscretizerShrinkByInflection(ExtraTreeClassifierParameterProxy, BaseDecisionTreeDiscretizerShrinkByInflection):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None, n_inflections=1,
                criterion="gini",
                splitter="random",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0,
                max_features="auto",
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                class_weight=None,
                ccp_alpha=0.0):
        BaseDecisionTreeDiscretizerShrinkByInflection.__init__(self, n_bins=n_bins, n_jobs=n_jobs, n_inflections=n_inflections)
        ExtraTreeClassifierParameterProxy.__init__(
                                                    self,
                                                    criterion=criterion,
                                                    splitter=splitter,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    class_weight=class_weight,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split,
                                                    random_state=random_state, 
                                                    ccp_alpha=ccp_alpha)


class ExtraTreeRegressorDiscretizerShrinkByInflection(ExtraTreeRegressorParameterProxy, BaseDecisionTreeDiscretizerShrinkByInflection):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, n_jobs=None, n_inflections=1,
                criterion="mse",
                splitter="random",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                random_state=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                max_leaf_nodes=None,
                ccp_alpha=0.0):
        BaseDecisionTreeDiscretizerShrinkByInflection.__init__(self, n_bins=n_bins, n_jobs=n_jobs, n_inflections=n_inflections)
        ExtraTreeRegressorParameterProxy.__init__(
                                                    self,
                                                    criterion=criterion,
                                                    splitter=splitter,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split,
                                                    random_state=random_state,
                                                    ccp_alpha=ccp_alpha)


DecisionTreeDiscretizer = DecisionTreeClassifierDiscretizer
ExtraTreeDiscretizer = ExtraTreeClassifierDiscretizer
DecisionTreeDiscretizerShrinkByInflection = DecisionTreeClassifierDiscretizerShrinkByInflection
ExtraTreeDiscretizerShrinkByInflection = ExtraTreeClassifierDiscretizerShrinkByInflection
