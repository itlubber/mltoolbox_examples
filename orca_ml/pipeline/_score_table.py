from collections import defaultdict
from pandas import DataFrame
from ._validation import check_classical_scorecard_pipeline
from ._validation import isa_imputer, isa_discretizer, isa_woe_encoder, isa_selector
from ..preprocessing.discretization._binning_table import get_bin_repr


def get_score_table(ppl, X, y=None, *, precision=2):
    """Get score table from a classical scorecard pipeline.

    The classical scorecard pipeline must have a discretizer, woe encoder, logistic regression and a score transformer.

    Parameters
    -----------
    ppl : Pipeline, a classical scorecard pipeline
    X : array-like, the train data
    y : array-like, default is None, the train label
    precision : int, bin interval precision

    Returns
    -----------
    A score table representing the pipeline scoring processing.
    """
    ppl = check_classical_scorecard_pipeline(ppl)
    imputer_map = {}
    discretizer_map = {}
    encoder_map = {}
    final_features = X.columns.values
    for idx, _, transformer in ppl._iter(with_final=False, filter_passthrough=False):
        Xt = transformer.transform(X)
        if isa_imputer(transformer):
            imputer_map = {k: v for k, v in zip(X.columns, transformer.statistics_)}
        elif isa_discretizer(transformer):
            closed = transformer.closed
            n_bins = transformer.n_bins_
            bin_edges = transformer.bin_edges_
            discretizer_map = {
                k: [
                    get_bin_repr(i, l, r, closed=closed, precision=precision)
                    for i, l, r in zip(range(n_bin), bin_edge[:-1], bin_edge[1:])
                ]
                for k, n_bin, bin_edge in zip(X.columns, n_bins, bin_edges)
            }
        elif isa_woe_encoder(transformer):
            numbers = transformer.numbers_
            encoder_map = {k: nums for k, nums in zip(X.columns, numbers)}
        elif isa_selector(transformer):
            selected_features = transformer.get_selected_features(X)
            if imputer_map:
                imputer_map = {k: imputer_map[k] for k in selected_features}
            if discretizer_map:
                discretizer_map = {k: discretizer_map[k] for k in selected_features}
            if encoder_map:
                encoder_map = {k: encoder_map[k] for k in selected_features}
            final_features = selected_features
        X = Xt
    
    scorecard = ppl._final_estimator
    lr, score_transformer = scorecard.classifier_, scorecard.transformer_
    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]
    A, B = score_transformer.A_, score_transformer.B_
    basic_score = A - B * intercept
    n_features = len(final_features)
    average_score = basic_score / n_features
    d = defaultdict(list)
    for i, feature, coef in zip(range(n_features), final_features, coefs):
        bin_reprs = discretizer_map[feature]
        woes = encoder_map[feature]
        n_bin = len(bin_reprs)
        d["feature index"].extend(i + 1 for _ in range(n_bin))
        d["feature name"].extend(feature for _ in range(n_bin))
        d["bin"].extend(bin_reprs)
        d["woe"].extend(woes)
        d["coefficient"].extend(coef for _ in range(n_bin))
        d["score"].extend(-w * coef * B + average_score for w in woes)
    
    df = DataFrame.from_dict(d)
    # names = ["feature index", "feature name", "bin", "ное", "coefficient", "score"]
    # index = MultiIndex.from_tuples(zip(*(d[f] for f in names)), names=names)
    # df = DataFrame(None, index=index)
    return df
