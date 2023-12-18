from sklearn.utils.validation import _deprecate_positional_args
from ._base import BaseSelectFromModel
from ...linear_model._logistic import LogisticRegressionParameterProxy


class SelectFromLR(LogisticRegressionParameterProxy, BaseSelectFromModel):
    """
    Examples
    >>> import pandas as pd
    >>> from orca_ml.feature_selection.from_model._Ir import SelectFromLR
    >>> X = pd.DataFrame([[0.87, -1.34, 0.31], [-2.79, -0.02, -0.85], [-1.34, -0.48, -2.55], [1.92, 1.48, 0.65]], columns=["f1", "f2", "f3"])
    >>> y = pd.Series([0, 1, 0, 1])
    >>> selector = SelectFromLR(penalty='12', C=1.0, threshold='mean').fit(X, y)
    >>> selector.transform(X)
         f2
    0 -1.34
    1 -0.02
    2 -0.48
    3  1.48
    """
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1,
                top_k=None, importance_getter='auto', l1_ratio=None,
                penalty='12', dual=False, tol=1e-4, C=1.0,
                fit_intercept=True, intercept_scaling=1, class_weight=None,
                random_state=None, solver='lbfgs', max_iter=100,
                multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                ):
        LogisticRegressionParameterProxy.__init__(self, class_weight=class_weight,
                                                penalty=penalty, dual=dual, tol=tol, C=C,
                                                fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                                random_state=random_state, solver=solver, max_iter=max_iter,
                                                multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                                n_jobs=n_jobs, l1_ratio=l1_ratio)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)
