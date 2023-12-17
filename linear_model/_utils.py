from scipy import sparse


def _rescale_data(X, weights):
    if sparse.issparse(X):
        size = weights.shape[0]
        weight_dia = sparse.dia_matrix((weights, 0), (size, size))
        X_rescaled = X * weight_dia
    else:
        X_rescaled = X * weights
    return X_rescaled
