import numbers
import warnings
from abc import abstractmethod
import numpy as np
from pandas import DataFrame
from sklearn import clone
from sklearn.tree._tree import TREE_UNDEFINED
from sklearn.utils import check_array
from sklearn.utils.validation import _deprecate_positional_args

from ._base import BaseRuleExtractor
from ._rule import Rule
from ..cluster import KMeansParameterProxy
from ..ensemble._forest import ExtraTreesClassifierParameterProxy, RandomForestClassifierParameterProxy
from ..tree import DecisionTreeClassifierParameterProxy, ExtraTreeClassifierParameterProxy


_MIN_BIN_WIDTH = 1e-8


def _gen_rule(tree, name, idx=0):
    if tree.feature[idx] != TREE_UNDEFINED:
        threshold = tree.threshold[idx]
        left, right = tree.children_left, tree.children_right

        # if left is not empty
        if left[idx] != -1:
            yield Rule(f"{name} <= {threshold}")
        yield from _gen_rule(tree, name, left[idx])

        # if right is not empty
        if right[idx] != -1:
            yield Rule(f"{name} > {threshold}")
        yield from _gen_rule(tree, name, right[idx])


class BaseUnivariateRuleExtractor(BaseRuleExtractor):
    def _validate_n_bins(self, n_features):
        """Returns n_bins_, the number of bins per feature."""
        orig_bins = self.n_bins
        if isinstance(orig_bins, numbers.Number):
            if not isinstance(orig_bins, numbers.Integral):
                raise ValueError("{} received an invalid n_bins type. Received {}, expected int.".format(self.__class__.__name__, type(orig_bins).__name__))
            if orig_bins < 2:
                raise ValueError("{} received an invalid number of bins. Received {}, expected at least 2.".format(self.__class__.__name__, orig_bins))
            return np.full(n_features, orig_bins, dtype=int)
    
        n_bins = check_array(orig_bins, dtype=int, copy=True, ensure_2d=False)

        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array of shape (n_features,).")

        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid number of bins at indices {}. Number of bins must be at least 2, and must be an int.").format(self.__class__.__name__, indices)
        return n_bins

    def fit(self, X, y=None, **fit_params):
        arr = check_array(X, dtype="numeric", estimator=self)
        if y is not None:
            y = self._check_y(y)
        
        _, n_features = X.shape
        n_bins = self._validate_n_bins(n_features)
        if isinstance(X, DataFrame):
            feature_names = X.columns.values
        else:
            feature_names = np.fromiter((f"f{i}" for i in range(n_features)), dtype=object)
        self.rules_ = [self._bin_one_column_and_extract_rules(i, feature_names[i], n_bins[i], arr[:, i], y=y) for i in range(n_features)]
        
        return self
    
    def _bin_one_column_and_extract_rules(self, i, name, n_bin, x, y=None, **kwargs):
        bin_edge = self._bin_one_column(i, n_bin, x, y=y, **kwargs)
        return self._make_rule_from_bin_edge(name, bin_edge)
    
    @abstractmethod
    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        raise NotImplementedError

    def _make_rule_from_bin_edge(self, name, bin_edge):
        if len(bin_edge) <= 2:
            return []
        bin_edge = bin_edge[1:-1]
        if self.closed == 'right':
            op1, op2 = '<=', '>'
        else:
            op1, op2 = '>=', '<'
        return [*(Rule("{} {} {}".format(name, op1, e)) for e in bin_edge),
                *(Rule("{} {} {}".format(name, op2, e)) for e in bin_edge)]


class BaseDecisionTreeRuleExtractor(BaseUnivariateRuleExtractor):
    _closed = "right"
    
    def _make_estimator(self):
        self.estimator = None
    
    def fit(self, X, y=None, **fit_params):
        self._make_estimator()
        return super(BaseDecisionTreeRuleExtractor, self).fit(X, y=y, **fit_params)

    def _bin_one_column(self, i, n_bin, x, y=None,**kwargs):
        col_min, col_max = x.min(), x.max()
        if col_min == col_max:
            warnings.warn("Feature {} is constant and will not produce any rule.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            return bin_edge
        
        dt = clone(self.estimator)
        dt.max_leaf_nodes = n_bin
        dt.fit(x[:, None], y)
        thresholds = sorted(s for s in dt.tree_.threshold if s != TREE_UNDEFINED)
        bin_edge = np.r_[col_min, thresholds, col_max]

        # Remove bins whose width are too small (i.e., <= 1e-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
        bin_edge = bin_edge[mask]

        if len(bin_edge) -1 != n_bin:
            warnings.warn('Bins vhose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
        
        return bin_edge


class DecisionTreeRuleExtractor(DecisionTreeClassifierParameterProxy, BaseDecisionTreeRuleExtractor):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, criterion="gini", splitter="best", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, class_weight="balanced", ccp_alpha=0.0):
        DecisionTreeClassifierParameterProxy.__init__(self, criterion=criterion, splitter=splitter, max_depth=max_depth,
                                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                      min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                      random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                      min_impurity_split=min_impurity_split, class_weight=class_weight, ccp_alpha=ccp_alpha)
        BaseDecisionTreeRuleExtractor.__init__(self, n_bins=n_bins)


class ExtraTreeRuleExtractor(ExtraTreeClassifierParameterProxy, BaseDecisionTreeRuleExtractor):
    @_deprecate_positional_args
    def __init__(self, *, n_bins=5, criterion="gini", splitter="random", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.,
                 min_impurity_split=None, class_weight=None, ccp_alpha=0.0):
        ExtraTreeClassifierParameterProxy.__init__(self, criterion=criterion, splitter=splitter, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, class_weight=class_weight, ccp_alpha=ccp_alpha)
        BaseDecisionTreeRuleExtractor.__init__(self, n_bins=n_bins)


class KMeansRuleExtractor(KMeansParameterProxy, BaseUnivariateRuleExtractor):
    def __init__(self, *, n_bins=5, init='uniform', n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=None, copy_x=True, algorithm='auto'):
        BaseUnivariateRuleExtractor.__init__(self, n_bins=n_bins)
        KMeansParameterProxy.__init__(self, init=init, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose,
                                      random_state=random_state, copy_x=copy_x, algorithm=algorithm)

    def fit(self, X, y=None, **fit_params):
        self._make_estimator()
        return super(KMeansRuleExtractor, self).fit(X, y=y, **fit_params)
    
    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        col_min, col_max = x.min(), x.max()
        if col_min == col_max:
            warnings.warn("Feature {} is constant and will not produce any rule.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            return bin_edge
        
        # Deterministic initialization with uniform spacing
        if self.init == 'uniform':
            uniform_edges = np.linspace(col_min, col_max, n_bin +1)
            init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
        else:
            init = self.init
        
        # 1D k-means procedure
        km = clone(self.estimator)
        km.n_clusters = n_bin
        km.init = init
        centers = km.fit(x[:, None]).cluster_centers_[:, 0]
        # Must sort, centers may be unsorted even with sorted init
        centers.sort()
        bin_edge = (centers[1:] + centers[:-1]) * 0.5
        bin_edge = np.r_[col_min, bin_edge, col_max]

        # Remove bins whose width are too small (i.e., <= le-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) >_MIN_BIN_WIDTH
        bin_edge = bin_edge[mask]

        if len(bin_edge) - 1 != n_bin:
            warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
        
        return bin_edge


class UniformRuleExtractor(BaseUnivariateRuleExtractor):
    def _bin_one_column( self, i, n_bin, x, y=None, **kwargs):
        col_min, col_max = x.min(), x.max()

        if col_min == col_max:
            warnings.warn("Feature {} is constant and will not produce any rule.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            return bin_edge
        
        bin_edge = np.linspace(col_min, col_max, n_bin + 1)

        # Remove bins whose width are too small (i.e., <= 1e-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
        bin_edge = bin_edge[mask]

        if len(bin_edge) - 1 != n_bin:
            warnings.warn('Bins whose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
        
        return bin_edge


class QuantileRuleExtractor(BaseUnivariateRuleExtractor):
    def _bin_one_column(self, i, n_bin, x, y=None, **kwargs):
        col_min, col_max = x.min(), x.max()

        if col_min == col_max:
            warnings.warn("Feature {} is constant and will not produce any rule.".format(i))
            bin_edge = np.array([-np.inf, np.inf])
            return bin_edge
        
        quantiles = np.linspace(0, 100, n_bin + 1)
        bin_edge = np.asarray(np.percentile(x, quantiles))

        # Remove bins whose width are too small (i.e., <= 1e-8)
        mask = np.ediff1d(bin_edge, to_begin=np.inf) > _MIN_BIN_WIDTH
        bin_edge  = bin_edge[mask]

        if len(bin_edge) - 1 != n_bin:
            warnings.warn('Bins vhose width are too small (i.e., <= 1e-8) in feature {} are removed. Consider decreasing the number of bins.'.format(i))
        
        return bin_edge


# Multivariate rule extractor
class BaseMultivariateRuleExtractor(BaseRuleExtractor):
    pass


class BaseMultivariateDecisionTreeRuleExtractor(BaseMultivariateRuleExtractor):
    def _validate_n_bins(self):
        """Returns n_bins_, the number of bins per feature."""
        orig_bins = self.n_bins
        if isinstance(orig_bins, numbers.Number):
            if not isinstance(orig_bins, numbers.Integral):
                raise ValueError("{} received an invalid n_bins type. Received {}, expected int.".format(self.__class__.__name__, type(orig_bins).__name__))
        if orig_bins < 2:
            raise ValueError("{} received an invalid number of bins. Received {}, expected at least 2.".format(self.__class__.__name__, orig_bins))
        
        return orig_bins

    @abstractmethod
    def _make_estimator(self):
        self.estimator = None

    def fit(self, X, y=None, **fit_params):
        _, n_features = X.shape
        n_bins = self._validate_n_bins()
        if isinstance(X, DataFrame):
            feature_names = X.columns.values
        else:
            feature_names = np.fromiter((f"f{i}" for i in range(n_features)), dtype=object)

        self._make_estimator()
        self.estimator.max_leaf_nodes = n_bins
        self.estimator.fit(X, y)
        estimators = self.estimator.estimators_
        n_estimators = len(estimators)
        self.rules_ = [_extract_one_tree(i, estimators[i], feature_names) for i in range(n_estimators)]
        return self


def _extract_one_tree(i, estimator, feature_names):
    rule =[]
    rules = []
    _gen_rule_for_one_tree(estimator.tree_, feature_names, rule, rules)
    return rules


def _gen_rule_for_one_tree(tree, feature_names, rule, rules, idx=0):
    if tree.feature[idx] != TREE_UNDEFINED:
        feature = feature_names[tree.feature[idx]]
        threshold = tree.threshold[idx]

        left, right = tree.children_left, tree.children_right

        # if left is not empty
        if left[idx] != -1:
            rule.append(f"({feature} <= {threshold})")
            _gen_rule_for_one_tree(tree, feature_names, rule, rules, left[idx])
            rule.pop()
        
        # if right is not empty
        if right[idx] != -1:
            rule.append(f"({feature} > {threshold})")
            _gen_rule_for_one_tree(tree, feature_names, rule, rules, right[idx])
            rule.pop()
    else:
        rules.append(Rule(' & '.join(rule)))


class RandomForestRuleExtractor(RandomForestClassifierParameterProxy, BaseMultivariateDecisionTreeRuleExtractor):
    def __init__(self, *, n_bins=5, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., n_jobs=None,
                 min_impurity_split=None, random_state=None, verbose=0, class_weight=None, ccp_alpha=0.0, max_samples=None):
        RandomForestClassifierParameterProxy.__init__(self, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                      min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                      max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                      min_impurity_split=min_impurity_split, n_jobs=n_jobs, random_state=random_state,
                                                      verbose=verbose, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        BaseMultivariateRuleExtractor.__init__(self, n_bins=n_bins)


class ExtraTreesRuleExtractor(ExtraTreesClassifierParameterProxy, BaseMultivariateDecisionTreeRuleExtractor):
    def __init__(self, *, n_bins=5, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., n_jobs=None, 
                 min_impurity_split=None, random_state=None, verbose=0, class_weight=None, ccp_alpha=0.0, max_samples=None):
        ExtraTreesClassifierParameterProxy.__init__(self, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split, n_jobs=n_jobs, random_state=random_state,
                                                    verbose=verbose, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)
        BaseMultivariateRuleExtractor.__init__(self, n_bins=n_bins)
