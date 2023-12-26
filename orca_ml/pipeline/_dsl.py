from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from ..compose import make_column_transformer
from ._base import FeatureUnion, make_pipeline, make_union
from ._random_choice import RandomChoice, make_random_choice
from ._validation import isa_composed, isa_pipeline, isa_feature_union, isa_column_transformer, isa_random_choice


_PIPELINE_SEP = '>>'
_UNION_SEP = '&'
_CHOICE_SEP = '|'
_COLUMN_TRANSFORMER_SEP = "+"
_ON = "@"
_WITH_PROB = "*"


def _get_class_name(estimator):
    """Get object's class name

    Parameters
    -----------
    estimator : Any
        Any object with a class name

    Returns
    -----------
    out : str
        The obiect's class name
    """
    return estimator.__class__.__name__


def dsl_repr(composed):
    """Get the dsl representation of a composed object

    Parameters
    -----------
    composed : instance of Pipeline or FeatureUnion
        Can be nested
    
    Returns
    -----------
    repr : str
        The topology representation of the composed object
    
    Examples
    -----------
    >>> from sklearn.pipeline import Pipeline, FeatureUnion
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from sklearn.decomposition import TruncatedsVD, PCA
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from orca_ml.feature_selection import CorrSelector, SelectFromLR
    >>> ppl = Pipeline([('a', OneHotEncoder()), 
                        ('b', FeatureUnion([("pca", PCA(n_components=1)), ("svd", TruncatedsVD(n_components=2))])),
                        ('c', FeatureUnion([
                            ("scale_ppl", Pipeline([("s1", MinMaxScaler()), ("sel", SelectFromLR())])),
                            ("svd", TruncatedsVD(n_components=2)),
                            ("corr", CorrSelector())
                            ])
                        )])
    >>> print(dsl_repr(ppl))
    (OneHotEncoder() >> (PCA(n_components=1) & TruncatedsVD()) >> ((MinMaxScaler() >> SelectFromLR()) & TruncatedsVD() & CorrSelector()))
    """
    if isa_pipeline(composed):
        sep = f" {_PIPELINE_SEP} "
        res = sep.join(dsl_repr(node) if isa_composed(node) else repr(node) for _, node in composed.steps)
    elif isa_feature_union(composed):
        sep = f" {_UNION_SEP} "
        res = sep.join(dsl_repr(node) if isa_composed(node) else repr(node) for _, node in composed.transformer_list)
    elif isa_column_transformer(composed):
        sep = f" {_COLUMN_TRANSFORMER_SEP} "
        res = sep.join(f"{dsl_repr(node)} {_ON}) {repr(columns)}" if isa_composed(node) else f"{repr(node)} {_ON} {repr(columns)}"
        for _, node, columns in composed.transformers)
        res += f"{_COLUMN_TRANSFORMER_SEP} {repr(composed.remainder)} {_COLUMN_TRANSFORMER_SEP} {repr(composed.sparse_threshold)}"
    elif isa_random_choice(composed):
        sep =" {} ".format(_CHOICE_SEP)
        if composed.p is None:
            res = sep.join(dsl_repr(node) if isa_composed(node) else repr(node)
                           for _, node in composed.candidates)
        else:
            res = sep.join(dsl_repr(node) if isa_composed(node) else f"{repr(node)} {_WITH_PROB} {prob}"
                           for (_, node), prob in zip(composed.candidates, composed.p))
    else:
        raise ValueError("Input should be a composed object!")
    return "({})".format(res)


def _and(self, other):
    if isa_feature_union(self):
        estimators1 = [step[-1] for step in self.transformer_list]
        if isa_feature_union(other):
            estimators2 = [step[-1] for step in other.transformer_list]
        else:
            estimators2 = [other]
        return make_union(*(estimators1 + estimators2))
    else:
        estimators1 = [self]
        if isa_feature_union(other):
            estimators2 = [step[-1] for step in other.transformer_list]
        else:
            estimators2 = [other]
        return make_union(*(estimators1 + estimators2))


def _rand(self, other):
    return _and(other, self)


def _rshift(self, other):
    if isa_pipeline(self):
        estimators1 = [step[-1] for step in self.steps]
        if isa_pipeline(other):
            estimators2 = [step[-1] for step in other.steps]
        else:
            estimators2 = [other]
        return make_pipeline(*(estimators1 + estimators2))
    else:
        estimators1 = [self]
        if isa_pipeline(other):
            estimators2 = [step[-1] for step in other.steps]
        else:
            estimators2 = [other]
        return make_pipeline(*(estimators1 + estimators2))


def _rrshift(self, other):
    return _rshift(other, self)


def _or(self, other):
    if isa_random_choice(self):
        estimators1 = [step[-1] for step in self.candidates]
        if isa_random_choice(other):
            estimators2 = [step[-1] for step in other.candidates]
        else:
            estimators2 = [other]
        return make_random_choice(*(estimators1 + estimators2))
    else:
        estimators1 = [self]
        if isa_random_choice(other):
            estimators2 = [step[-1] for step in other.candidates]
        else:
            estimators2 = [other]
        return make_random_choice(*(estimators1 + estimators2))


def _ror(self, other):
    return _or(other, self)


BaseEstimator.__and__ = _and
BaseEstimator.__rand__ = _rand
BaseEstimator.__rshift__ = _rshift
BaseEstimator.__rrshift__ = _rrshift
BaseEstimator.__or__ = _or
BaseEstimator.__ror__ = _ror


class _TransformerOnColumns:
    def __init__(self, transformer, columns):
        self._transformer = (transformer, columns)
        
    def __add__(self, other):
        if isinstance(other, ColumnTransformer):
            transformers = [(trans, cols) for _, trans, cols in other.transformers]
            return make_column_transformer(self._transformer, *transformers)
        elif isinstance(other, _TransformerOnColumns):
            return make_column_transformer(self._transformer, other._transformer)
        elif isinstance(other, str):
            return make_column_transformer(self._transformer, remainder=other)
        elif isinstance(other, float):
            return make_column_transformer(self._transformer, sparse_threshold=other)
        else:
            raise PipelineSyntaxError("_TransformerOnColumns should be added to another _TransformerOnColumns or ColumnTransformer, string, float number.")
    
    def __radd__(self, other):
        if isinstance(other, ColumnTransformer):
            transformers = [(trans, cols) for _, trans, cols in other.transformers]
            return make_column_transformer(*transformers, self._transformer)
        elif isinstance(other, _TransformerOnColumns):
            return make_column_transformer(other._transformer, self._transformer)
        elif isinstance(other, str):
            return make_column_transformer(self._transformer, remainder=other)
        elif isinstance(other, float):
            return make_column_transformer(self._transformer, sparse_threshold=other)
        else:
            raise PipelineSyntaxError("_TransformerOnColumns should be added to another _TransformerOnColumns or ColumnTransformer, string, float number.")


def _transformer_columns_pair(self, columns):
    return _TransformerOnColumns(self, columns)


BaseEstimator.__truediv__ = _transformer_columns_pair
BaseEstimator.__matmul__ = _transformer_columns_pair


def _column_transformer_add(self, other):
    if isinstance(other, ColumnTransformer):
        transformers1 = [(trans, cols) for _, trans, cols in self.transformers]
        transformers2 = [(trans, cols) for _, trans, cols in other.transformers]
        return make_column_transformer(*transformers1, *transformers2)
    elif isinstance(other, _TransformerOnColumns):
        transformers = [(trans, cols) for _, trans, cols in self.transformers]
        return make_column_transformer(*transformers, other._transformer)
    elif isinstance(other, BaseEstimator):
        if hasattr(other, 'fit') and hasattr(other, 'transform'):
            self.remainder = other
            return self
        else:
            raise ValueError("Remainder estimator should support 'fit' and 'transform'.")
    elif isinstance(other, str):
        self.remainder = other
        return self
    elif isinstance(other, float):
        self.sparse_threshold = other
        return self
    else:
        raise PipelineSyntaxError


ColumnTransformer.__add__ = _column_transformer_add


class _EstimatorWithProb:
    def __init__(self, estimator, prob):
        self._candidate = (estimator, prob)
    
    def __or__(self, other):
        if isinstance(other, RandomChoice):
            _, candidates = zip(*other.candidates)
            candidates = [self._candidate[0], *candidates]
            p = [self._candidate[1], *other.p]
            return make_random_choice(*candidates, p=p)
        elif isinstance(other, _EstimatorWithProb):
            candidates, p = zip(*[self._candidate, other._candidate])
            return make_random_choice(*candidates, p=p)
        else:
            raise PipelineSyntaxError("_EstimatorWithProb should be coupled with another _EstimatorWithProb or RandomChoice.")
    
    def __ror__(self, other):
        if isinstance(other, RandomChoice):
            _, candidates = zip(*other.candidates)
            candidates = [*candidates, self._candidate[0]]
            p = [*other.p, self._candidate[1]]
            return make_random_choice(*candidates, p=p)
        elif isinstance(other, _EstimatorWithProb):
            candidates, p = zip(*[other._candidate, self._candidate])
            return make_random_choice(*candidates, p=p)
        else:
            raise PipelineSyntaxError("_EstimatorWithProb should be coupled with another _EstimatorWithProb or RandomChoice.")


def _estimator_prob_pair(self, prob):
    return _EstimatorWithProb(self, prob)


BaseEstimator.__mul__ = _estimator_prob_pair


class PipelineSyntaxError(SyntaxError):
    pass
