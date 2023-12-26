import numpy as np
from scipy.stats import pearsonr as _pearsonr
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from _univariate_selection import _BaseFilter, score_func_map_for_classification, score_func_map_for_regression


def pearsonr(X, y):
    X = check_X_y(X, y, dtype="numeric", y_numeric=True, force_all_finite=True, ensure_2d=True)
    _, n_features = X.shape
    scores, pvalues = zip(*(_pearsonr(X[:, i], y) for i in range(n_features)))
    # Alternatively, you can just return scores and pvalues as tuple
    return np.asarray(scores), np.asarray(pvalues)


extra_score_func_map = {
    'pearsonr': pearsonr
}


score_func_map_for_classification.update(extra_score_func_map)
score_func_map_for_regression.update(extra_score_func_map)


class SelectCorrelation(_BaseFilter):

    def __init__(self, score_func=pearsonr, *, threshold=0.8):
        super(SelectCorrelation, self).__init__(score_func=score_func)
        self.threshold = threshold

    def _get_support_mask(self):
        check_is_fitted(self, "scores_")
        threshold = _calculate_threshold(self, self.scores_, self.threshold)
        return self.scores_ > threshold
