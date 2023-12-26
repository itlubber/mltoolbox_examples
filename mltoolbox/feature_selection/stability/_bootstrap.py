"""Bootstrap helper functions This module contains helper function for stability selection that do bootstrap sampling."""

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.random import sample_without_replacement


def bootstrap_without_replacement(y, n_subsamples, random_state=None):
    """Bootstrap without replacement, irrespective of label.

    It is a wrapper around sklearn.utils.random.sample_without_replacement.

    Parameters
    ----------
    y : array of size (n_subsamples, )
        True labels
    n_subsamples : int
        Number of subsamples in the bootstrap sampling.
    random_state : int, np.random.RandomState or None, default = None
        Pseudo random generator state used for random uniform sampling from lists of possible values instead of scipy. stats distributions
        If int, random_state is the seed used by the random number generator;
        If Randomstate instance, random_state is the random number generator;
        If None, the random number generator is the Randomstate instance used by np.random.

    Returns
    ----------
    out : array-like of shape (n_subsamples, )
        The sampled subsets of integer. The subset of selected integer might not be randomized, see the method argument.
    """
    n_samples = y.shape[0]
    return sample_without_replacement(n_samples, n_subsamples, random_state=random_state)


def complementary_pairs_bootstrap(y, n_subsamples, random_state=None):
    """Complementary pairs bootstrap.

    Two disjoint subsamples A and B are generated, such that
    |A| = n_subsamples, the union of A and B equals {0, . .., n_samples - 1}. Samples irrespective of label.

    Parameters
    ----------
    y : array of size (n_subsamples, )
        True labels
    n_subsamples : int
        Number of subsamples in the bootstrap sampling.
    random_state : int, np.random.RandomState or None, default = None
        Pseudo random generator state used for random uniform sampling from lists of possible values instead of scipy.stats distributions
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random

    Returns
    ----------
    out : array-like of shape (n_subsamples, )
        The sampled subsets of integer. The subset of selected integer might not be randomized, see the method argument.
    B: array-like of shape (n_subsamples, )
        The complement of A.
    """
    n_samples = y.shape[0]
    subsample = sample_without_replacement(n_samples, n_subsamples, random_state=random_state)
    complementary_subsample = np.setdiffld(np.arange(n_samples), subsample)
    return subsample, complementary_subsample


def stratified_bootstrap(y, n_subsamples, random_state=None):
    """Bootstrap without replacement, performed seperately for each group in y.

    Parameters
    ----------
    y : array-like of shape (n_samples, 
        The label
    n_subsamples : int
        Number of subsamples in the bootstrap sampling
    random_state : int, np.random.RandomState or None, default = None
        Pseudo random generator state used for random uniform sampling from lists of possible values instead of scipy.stats distributions
        If int, random_state is the seed used by the random number generator;
        If Randomstate instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random

    Returns
    ----------
    out : array-like of shape (n_subsamples, )
        The sampled subsets of integer. The subset of selected integer might not be randomized, see the method argument.
    """
    target_type = type_of_target(y)
    allowed_target_types = ('binary', 'multiclass')
    if target_type not in allowed_target_types:
        raise ValueError("Supported target type are: {}. Got {!r} instead. ". format(allowed_target_types, target_type))
    unique_y, y_counts = np.unique(y, return_counts=True)
    y_n_samples = np.int64(np.round(y_counts / y_counts.sum() * n_subsamples))

    # the above should return grouped subsamples which approximately sum uр to n_subsamples but may not work out exactly due to rounding errors
    # If this is the case, adjust the count of the largest class.
    if y_n_samples.sum() != n_subsamples:
        delta = n_subsamples - y_n_samples.sum()
        y_n_samples[np.argmax(y_counts)] += delta
    
    def _gen(classes, class_samples):
        for cls, cls_n_samples in zip(classes, class_samples):
            indices = np.where(y == cls)[0]
            n_samples = len(indices)
            yield from indices[sample_without_replacement(n_samples, cls_n_samples, random_state=random_state)]
    
    all_selected = np.fromiter((i for i in _gen(unique_y, y_n_samples)), dtype=np.int64)
    return all_selected
