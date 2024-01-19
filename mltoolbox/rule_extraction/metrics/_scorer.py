import warnings
import numpy as np
from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics._scorer import check_scoring, make_scorer, get_scorer, _check_multimetric_scoring


def support_score(y_true, y_pred, *, pos_label=None, sample_weight=None):
    """Support is the fraction of the true value in predictions and target values.

    Parameters
    ------------
    y_true : array-like of shape (n_samples,)
        True class labels
    y_pred : array-like of shape (n_samples,)
        Predicted class labels
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-1ike of shape (n_samples,), default=None
        Sample weights

    Returns
    ------------
    score : float
        Support score in the range [0, 1]
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))
    
    check_consistent_length(y_true, y_pred, sample_weight)
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    assert_all_finite(y_true)
    assert_all_finite(y_pred)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified classes . dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a Futurewarning by calling np.array_equal(a, b) when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (classes.dtype.kind in ('O, U', 'S') or
        not (np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1]) or np.array_equal(classes, [0]) or np.array_equal(classes, [-1]) or np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and pos_label is not specified: either make y_true take value in {0, 1}} or {{-1, 1}} or pass pos_label explicitly.". format(classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    y_pred = (y_pred == pos_label)

    all_prod = np.column_stack((y_true, y_pred))
    support = np.count_nonzero(np.all(all_prod == 1, axis=1))
    if support == 0:
        warnings.warn("Zero support value encountered.")
    return support / float(y_true.shape[0])


support_scorer = make_scorer(support_score, greater_is_better=True)
