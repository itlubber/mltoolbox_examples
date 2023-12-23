from functools import partial
from typing import Generator, Iterator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from ._random_choice import RandomChoice
from .. compose import isa_column_transformer, isa_scorecard
from .. feature_selection._base import isa_selector
from ..impute import isa_imputer
from ..linear_model._logistic import LogisticRegression
from ..postprocessing import StandardScoreTransformer
from ..preprocessing.category_encoders import WoeEncoder
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
