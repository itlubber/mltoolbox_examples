from sklearn.utils.validation import _deprecate_positional_args
from ._base import BaseSelectFromModel
from ...ensemble._forest import RandomForestClassifierParameterProxy, RandomForestRegressorParameterProxy, ExtraTreesClassifierParameterProxy, ExtraTreesRegressorParameterProxy


class SelectFromRandomForestClassifier(RandomForestClassifierParameterProxy, BaseSelectFromModel):
    """
    Examples
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    >>> clf = SelectFromRandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, v)
    RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.transform(X)

    DecisionTreeClassifier, ExtraTreesClassifier
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1, top_k=None,
        importance_getter='auto',
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None
    ):
        RandomForestClassifierParameterProxy.__init__(self,
                                                        n_estimators=n_estimators,
                                                        criterion=criterion,
                                                        max_depth=max_depth,
                                                        min_samples_split=min_samples_split,
                                                        min_samples_leaf=min_samples_leaf,
                                                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                        max_features=max_features,
                                                        max_leaf_nodes=max_leaf_nodes,
                                                        min_impurity_decrease=min_impurity_decrease,
                                                        min_impurity_split=min_impurity_split,
                                                        bootstrap=bootstrap,
                                                        oob_score=oob_score,
                                                        n_jobs=n_jobs,
                                                        random_state=random_state,
                                                        verbose=verbose,
                                                        warm_start=warm_start,
                                                        class_weight=class_weight,
                                                        ccp_alpha=ccp_alpha,
                                                        max_samples=max_samples)
        BaseSelectFromModel.__init__(self, threshold=threshold, prefit=False, norm_order=norm_order, top_k=top_k, importance_getter=importance_getter)


class SelectFromExtraTreesClassifier(ExtraTreesClassifierParameterProxy, BaseSelectFromModel):
    @_deprecate_positional_args
    def __init__(self, *, threshold=None,
                norm_order=1,
                top_k=None,
                importance_getter='auto',
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                max_samples=None,
                ):
        ExtraTreesClassifierParameterProxy.__init__(self,
                                                    n_estimators=n_estimators,
                                                    criterion=criterion,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    min_impurity_split=min_impurity_split,
                                                    bootstrap=bootstrap,
                                                    oob_score=oob_score,
                                                    n_jobs=n_jobs,
                                                    random_state=random_state,
                                                    verbose=verbose,
                                                    warm_start=warm_start,
                                                    class_weight=class_weight,
                                                    ccp_alpha=ccp_alpha,
                                                    max_samples=max_samples)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)


class SelectFromRandomForestRegressor(RandomForestRegressorParameterProxy, BaseSelectFromModel):
    @_deprecate_positional_args
    def __init__(self, *, threshold=None,
                norm_order=1,
                top_k=None,
                importance_getter='auto',
                n_estimators=100,
                criterion="mse",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
    ):
        RandomForestRegressorParameterProxy.__init__(self,
                                                    n_estimators=n_estimators,
                                                    criterion=criterion,
                                                    max_depth=max_depth,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_samples_split=min_samples_split,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split,
                                                    bootstrap=bootstrap,
                                                    oob_score=oob_score,
                                                    random_state=random_state,
                                                    n_jobs=n_jobs,
                                                    verbose=verbose,
                                                    warm_start=warm_start,
                                                    ccp_alpha=ccp_alpha,
                                                    max_samples=max_samples)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)


class SelectFromExtraTreesRegressor(ExtraTreesRegressorParameterProxy, BaseSelectFromModel):
    @_deprecate_positional_args
    def __init__(self, *, threshold=None,
                norm_order=1,
                top_k=None,
                importance_getter='auto',
                n_estimators=100,
                criterion="mse",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.,
                min_impurity_split=None,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
                ):
        ExtraTreesRegressorParameterProxy.__init__(self,
                                                    n_estimators=n_estimators,
                                                    criterion=criterion,
                                                    max_depth=max_depth,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf,
                                                    min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    max_features=max_features,
                                                    max_leaf_nodes=max_leaf_nodes,
                                                    min_impurity_decrease=min_impurity_decrease,
                                                    min_impurity_split=min_impurity_split,
                                                    bootstrap=bootstrap,
                                                    oob_score=oob_score,
                                                    n_jobs=n_jobs,
                                                    random_state=random_state,
                                                    verbose=verbose,
                                                    warm_start=warm_start,
                                                    ccp_alpha=ccp_alpha,
                                                    max_samples=max_samples
                                                    )

        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)
