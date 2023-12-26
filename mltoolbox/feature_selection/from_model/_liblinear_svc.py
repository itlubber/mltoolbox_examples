from sklearn.utils.validation import _deprecate_positional_args

from ._base import BaseSelectFromModel
from ...svm._linear_svc import LinearSVCParameterProxy, LinearSVRParameterProxy


class SelectFromLibLinearSVC(LinearSVCParameterProxy, BaseSelectFromModel):
    # Liblinear is not deterministic as it uses a RNG inside
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1, top_k=None, importance_getter='auto',
                penalty='12', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True,
                intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000):
        LinearSVCParameterProxy.__init__(self,
                                        penalty=penalty, loss=loss, dual=dual, tol=tol,
                                        C=C, multi_class=multi_class, fit_intercept=fit_intercept,
                                        intercept_scaling=intercept_scaling, class_weight=class_weight,
                                        verbose=verbose,
                                        random_state=random_state, max_iter=max_iter)
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)


class SelectFromLibLinearSVR(LinearSVRParameterProxy, BaseSelectFromModel):
    # Liblinear is not deterministic as it uses a RNG inside
    @_deprecate_positional_args
    def __init__(self, *, threshold=None, norm_order=1, top_k=None, importance_getter='auto', epsilon=0.0, tol=1e-4, C=1.0,
            loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1., dual=True, verbose=0, random_state=None, max_iter=1000):
        LinearSVRParameterProxy.__init__(self,
                                        epsilon=epsilon, tol=tol, C=C,
                                        loss=loss, fit_intercept=fit_intercept,
                                        intercept_scaling=intercept_scaling, dual=dual, verbose=verbose,
                                        random_state=random_state, max_iter=max_iter)
        
        BaseSelectFromModel.__init__(self,
                                    threshold=threshold,
                                    prefit=False,
                                    norm_order=norm_order,
                                    top_k=top_k,
                                    importance_getter=importance_getter)
