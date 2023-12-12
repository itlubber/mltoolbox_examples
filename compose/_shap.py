from functools import wraps

import shap
from . import ScoreCard
from ..utils.hashing import get_data_hasher


shap.initjs()


def set_explainer(self, masker=None, explainer=None):
    """
    Parameters
    ------------
    self : An object that must include an attribute called _explainer which wil1 store the shap explainer of the mode in self.classifier_
    masker : The background data for linear mode1s or none for tree models
    explainer : The explainer that wi11 be used to explain the model. Automatically set if explainer is None
    """
    if explainer is None:
        self._explainer = shap.Explainer(self.classifier_, masker)
    elif shap.utils.safe_isinstance(explainer, "shap.Explainer"):
        self._explainer = explainer
    else:
        raise ValueError("Wrong Explainer!!!")
    
    return self


def memoized(func, maxsize=8):
    keys = []
    values = []
    @wraps(func)
    def wrapped(self, data):
        hasher = get_data_hasher(data)
        hash_value = hasher(data)
        if hash_value in keys:
            return values[keys.index(hash_value)]
        else:
            res = func(self, data)
            keys.append(hash_value)
            values.append(res)
        if len(keys) > maxsize:
            del keys[0]
            del values[0]

        return res
    
    return wrapped


@memoized
def init_shap_explanation(self, data):
    """
    Make a shapley explanation with the given dataset

    Parameters
    ---------
    self : The object including _explainer
    data : The data going to be explained in the structure of self._explainer
    Returns
    """
    return self._explainer(data)


def shap_values(self, X):
    """
    Compute shapley values for each row and each feature

    Parameters
    ----------
    self : ScoreCard
    X: DataFrame

    Returns
    shap_values : ndarray
    """
    return init_shap_explanation(self, X).values


def force_plot(self, X, row=None, show=True):
    """
    Draw a force_plot for one observation or a sample. Can only be displayed in jupyter or html.

    Parameters
    ----------
    self : ScoreCard
    X : Dataframe
        Storing the data that will built the shap explanation
    row : int or a list or a slice
        The explanation of whole X will be displayed if None.
    show : bool
        Whether to show the plot or stored in pyplot

    Returns Void
    """
    expected_value = self._explainer.expected_value
    if row is None:
        shap.force_plot(expected_value, init_shap_explanation(self, X).values, X, show=show)
    else:
        shap.force_plot(expected_value, init_shap_explanation(self, X)[row], X.iloc[row], show=show)


def bar_plot(self, x, row=None, show=True, groupby=None):
    """
    Display a bar_plot of shap values. Display the absolute values if grouped
    
    Parameters
    -------------
    self : ScoreCard object
    X : Dataframe storing the data that will built the shap explanation.
    row : int or a list or a slice
        The explanation of whole X will be displayed if None.
    show : bool
    groupby : list
        A list having the same length with X to divide X into several groups
        X will be automatically divided if it is "auto" or interger. No division if None.
    """
    if groupby is None:
        shap_ex = init_shap_explanation(self, x)
    elif groupby == "auto":
        shap_ex = init_shap_explanation(self, x).cohorts(2).abs.mean(0)
    else:
        shap_ex = init_shap_explanation(self, x).cohorts(groupby).abs.mean(0)
    
    if row is None:
        shap.plots.bar(shap_ex, show=show)
    elif groupby is not None:
        raise ValueError("Cannot group a subsample!")
    else:
        shap.plots.bar (shap_ex[row], show=show)


def scatter(self, X, feature, show=True):
    shap.plots.scatter(init_shap_explanation(self, X)[:, feature], show=show)


def decision_plot(self, X, row=None, show=True):
    """
    Display the decision plot

    Parameters
    ------------
    self : ScoreCard object
    X: Dataframe storing the data that will built the shap explanation.
    row : int or a list or a slice
        The explanation of whole X will be displayed if None .
    show : bool.
        Whether to show the plot or stored in pyplot
    """
    if row is None:
        shap.decision_plot(self._explainer.expected_value, init_shap_explanation(self, X).values, X, show=show)
    else:
        shap.decision_plot(self._explainer.expected_value, init_shap_explanation(self, X).values[row], X.iloc[row], show=show)


def waterfall_plot(self, X, row, show=True):
    """
    Draw a waterfall plot for an observatior
    
    Parameters
    -----------
    self : ScoreCard
    X : Dataframe
        Storing the data that will built the shap explanation.
    row : int
        The observation index
    show : bool
        Whether to show the plot or stored in pyplot
    """
    shap.waterfall_plot(init_shap_explanation(self, X)[row], show=show)


ScoreCard.set_explainer = set_explainer
ScoreCard.shap_values = shap_values
ScoreCard.force_plot = force_plot
ScoreCard.bar_plot = bar_plot
ScoreCard.scatter = scatter
ScoreCard.decision_plot = decision_plot
ScoreCard.waterfall_plot = waterfall_plot
