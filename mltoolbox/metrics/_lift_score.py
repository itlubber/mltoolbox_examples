import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing._label import _encode
from sklearn.utils import check_consistent_length, column_or_1d, check_array, assert_all_finite
from sklearn.utils._encode import _unique
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.multiclass import type_of_target


def lift_score(y_true, y_pred, pos_label=1, sample_weight=None):
    """Lift measures the degree to which the predictions of a
    classification model are better than randomly-generated predictions.

    The in terms of True Positives (TP), True Negatives (TN),
    False Positives (FP), and False Negatives (FN), the lift score is
    computed as:
    [ TP / (TP+FP) ] / [ (TP+FN) / (TP+TN+FP+FN) ]

    Parameters
    -----------
    y_true : array-like, shape=[n_samples]
        True class labels.
    y_pred : array-like, shape=[n_samples]
        Predicted class labels.
    binary : bool (default: True)
        Maps a multi-class problem onto a
        binary, where
        the positive class is 1 and
        all other classes are 0.
    pos_label : int (default: 0)
        Class label of the positive class.

    Returns
    ----------
    score : float
        Lift score in the range [0, infinity]

    Examples
    -----------
    >>> import numpy as np
    >>> from mltoolbox.metrics._ranking import lift_score
    >>> y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0])
    >>> lift_score(y_true, y_pred) # (4 / 6) / (6 / 9) = (4 / 9) / ((6 / 9) + (6 / 9))
    1.0

    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/
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
    
    # ensure binary classification if pos_Label is not specified classes.dtype.kind in ('o', 'u', 's') is required to avoid
    # triggering a Futurewarning by calling np.array_equal (a, b) when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
                classes.dtype.kind in ('O', 'U', 'S') or
                not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and pos_label is not specified: either make y_true take value in {e, 1}} or {{-1, 1}} or pass pos_label explicitly.".format(classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    y_pred = (y_pred == pos_label)
    return support(y_true, y_pred) / (support(y_true) * support(y_pred))


def support(y_target, y_predicted=None):
    """Support is the fraction of the true value in predictions and target values.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels.
    y_predicted : array-like, shape=[n_samples]
        Predicted class labels.

    Returns
    ----------
    score : float
        Support score in the range [0, 1]

    """
    if y_predicted is None:
        if y_target.ndim == 1:
            return (y_target == 1).sum() / float(y_target.shape[0])
        return (y_target == 1).all(axis=1).sum() / float(y_target.shape[0])
    else:
        all_prod = np.column_stack([y_target, y_predicted])
        return (all_prod == 1).all(axis=1).sum() / float(all_prod.shape[0])
