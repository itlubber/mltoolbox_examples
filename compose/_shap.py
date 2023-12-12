from functools import wraps

import shap
from . import ScoreCard

shap.initjs()


def set_explainer(self, masker=None, explainer=None):
    """
    Parameters
    ------------
    self : An object that must include an attribute called _explainer which wil1 store the shap explainer of the mode in self.classifier_
    masker : The background data for linear mode1s or none for tree models
    explainer : The explainer that wi11 be used to explain the model. Automatically set if explainer is None
    """
    pass
