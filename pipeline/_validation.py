from functools import partial
from typing import Generator, Iterator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from ._random_choice import RandomChoice
from ..compose import isa_column_transformer, isa_scorecard
from ..feature_selection._base import isa_selector
from ..impute._base import isa_imputer
from ..linear_model._logistic import LogisticRegression
from ..postprocessing import StandardScoreTransformer
from ..preprocessing.category_encoders._woe import WoeEncoder
from ..preprocessing.discretization import isa_discretizer


def _is_instance(typ, obj):
    return isinstance(obj, typ)


isa_estimator = partial(_is_instance, BaseEstimator)
isa_transformer = partial(_is_instance, TransformerMixin)
isa_pipeline = partial(_is_instance, Pipeline)
isa_feature_union = partial(_is_instance, FeatureUnion)
isa_random_choice = partial(_is_instance, RandomChoice)
isa_pipeline_or_union = partial(_is_instance, (Pipeline, FeatureUnion))
isa_composed = partial(_is_instance, (Pipeline, FeatureUnion, RandomChoice, ColumnTransformer))
isa_woe_encoder = partial(_is_instance, WoeEncoder)
isa_logistic_regression = partial(_is_instance, LogisticRegression)
isa_standard_score_transformer = partial(_is_instance, StandardScoreTransformer)


def is_pipeline(obj):
    return issubclass(obj, Pipeline) or isinstance(obj, Pipeline)


def is_union(obj):
    return issubclass(obj, FeatureUnion) or isinstance(obj, FeatureUnion)


def is_scorecard_pipeline(est):
    """Check whether an estimator is a ScoreCard Pipeline.

    Parameters
    -----------
    est : BaseEstimator, any sklearn-compatible estimator
    
    Returns
    -----------
    res : True if estimator is a ScoreCard Pipeline
    """
    if not isa_pipeline(est):
        return False
    final_estimator = est._final_estimator
    if not isa_scorecard(final_estimator):
        return False
    return True


def _get_all_estimators(composed):
    """Get all estimators inside a composed estimator.

    Parameters
    -----------
    composed : BaseEstimator, a sklearn-compatible estimator

    Returns
    -----------
    a single estimator or a list of nested estimators
    """
    if isa_pipeline(composed):
        return [_get_all_estimators(node) if isa_composed(node) else node for _, node in composed.steps]
    elif isa_feature_union(composed):
        return [_get_all_estimators(node) if isa_composed(node) else node for _, node in composed.transformer_list]
    elif isa_column_transformer (composed):
        return [_get_all_estimators(node) if isa_composed(node) else node for _, node, _ in composed.transformers]
    elif isa_random_choice(composed):
        return [_get_all_estimators(node) if isa_composed(node) else node for _, node in composed.candidates]
    elif isa_estimator(composed):
        return composed
    else:
        raise ValueError("Input should be a BaseEstimator!")


def _flatten(estimators):
    """Helper function for flatten

    Parameters
    -----------
    estimators : Iterator [BaseEstimator], a iterator of estimators

    Returns
    -----------
    estimator : Generator [BaseEstimator, None, None]
    """
    for est in estimators:
        if isinstance(est, list):
            yield from _flatten(est)
    else:
        yield est


def flatten(estimators):
    """Flatten a list of nested estimators

    Parameters
    -----------
    estimators : Iterator[BaseEstimator
    
    Returns
    -----------
    estimators : List[BaseEstimator], list of estimators
    """
    return [_ for _ in _flatten(estimators)]


def get_all_estimators(composed):
    """Get all estimators from a composed estimator

    Parameters
    -----------
    combosed : BaseEstimator, a composed estimator

    Returns
    -----------
    A flattened list of estimators
    """
    return flatten(_get_all_estimators(composed))


def check_classical_scorecard_pipeline(est):
    """Check if an estimator is a classical scorecard pipeline

    The classical scorecard pipeline must have a discretizer, ,woe encoder, logistic regression and a score transformer.

    Parameters
    -----------
    est : BaseEstimator, any sklearn-compatible estimator

    Returns
    -----------
    pipeline : Pipeline, a classical scorecard pipelines
    """
    if not isa_pipeline(est):
        raise ValueError("Input should be a Pipeline.")
    final_estimator = est._final_estimator
    if not isa_scorecard(final_estimator):
        raise ValueError("Input should be a ScoreCard Pipeline.")
    classifier = final_estimator.classifier
    if not isa_logistic_regression(classifier):
        raise ValueError("A classical scorecard pipeline requires LogisticRegression to be a classifier.")
    transformer = final_estimator.transformer
    if not isa_standard_score_transformer(transformer):
        raise ValueError("A classical scorecard pipeline requires StandardscoreTransformer to be a score transformer.")
    imputer_idx = -1
    discretizer_idx = -1
    woe_idx = -1
    for idx, _, trans in est._iter(with_final=False, filter_passthrough=False):
        if isa_imputer(trans):
            imputer_idx = idx
        elif isa_discretizer(trans):
            discretizer_idx = idx
        elif isa_woe_encoder(trans):
            woe_idx = idx
        else:
            if not isa_selector(trans):
                raise ValueError(f"A classical scorecard pipeline requires other estimator to be a SelectorMixin), but got {type(trans)}")
    if discretizer_idx == -1 or woe_idx == -1:
        raise ValueError("A classical scorecard pipeline requires Discretizer and WoeEncoder.")
    if imputer_idx != -1 and imputer_idx > woe_idx:
        raise ValueError("Imputer should be placed before WoeEncoder.")
    if discretizer_idx > woe_idx:
        raise ValueError("Discretizer should be placed before WoeEncoder.")
    return est
