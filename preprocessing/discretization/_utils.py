import numpy as np


def get_bin_repr(bin_index, left, right, *, closed='left', precision=2):
    if bin_index == -2:
        bin_repr = "missing"
    elif bin_index == -1:
        bin_repr = "special"
    else:
        if closed == 'right':
            repr_tp = f"({{:.{precision}f}}, {{:.{precision}f}}]"
        else:
            repr_tp = f"[{{:.{precision}f}}, {{: {precision}f}})"
        bin_repr = repr_tp.format(left, right)
    return bin_repr


def renormalize_bin_edges(bin_edges, input_features, selected_features):
    """Renormalize bin edges via selected features.

    Parameters
    ------------
    bin_edges : list of array of shape (n_features,)
        Usually comes from a discretizer's fit result.
    input_features : list of str of shape (n_features,)
        Usually comes from a discretizer's fit data.
    selected_features : list of str of shape (n_selected_features,)
        Usually comes from a selector's transform result.

    Returns
    ------------
    output_bin_edges : list of array of shape (n_selected_features,)
        List of selected features' bin edges.
    """
    if len(bin_edges) != len(input_features):
        raise ValueError("Inconsistent length of bin edges and input features.")
    unique_features = set(input_features)
    if len(input_features) != len(unique_features):
        raise ValueError("Input features should be unique.")
    if not all(f in input_features for f in selected_features):
        raise ValueError("Selected feature not in input features.")
    new_bin_edges = []
    for feature in selected_features:
        idx = np.flatnonzero(input_features == feature)[0]
        new_bin_edges.append(bin_edges[idx])
    return new_bin_edges
