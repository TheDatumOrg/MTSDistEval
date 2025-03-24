import numpy as np

from tslearn.metrics import dtw as dtw_tslearn

from numba import njit,prange

@njit(parallel=True,cache=True)
def dtw_d_dist_mat(X,Y,bounding_matrix):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    for p in prange(X.shape[0]):
        for q in range(Y.shape[0]):
            dist_mat[p,q] = dtw(X[p],Y[q],bounding_matrix) 
    return dist_mat

@njit(parallel=True,cache=True)
def dtw_i_dist_mat(X,Y,bounding_matrix):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    for p in prange(X.shape[0]):
        for q in range(Y.shape[0]):
            for c in range(X.shape[1]):
                x_channel_c = np.expand_dims(X[p,c],axis=0)
                y_channel_c = np.expand_dims(Y[q,c],axis=0)
                dist_mat[p,q] += dtw(x_channel_c,y_channel_c,bounding_matrix)
    return dist_mat

@njit(cache=True)
def dtw(x,y,bounding_matrix):
    cost_matrix = _cost_matrix(x, y, bounding_matrix)
    return cost_matrix[-1, -1]

@njit(cache=True)
def dtw_i(x,y,bounding_matrix):
    partial_dist = 0
    for c in range(x.shape[0]):
        partial_dist = partial_dist + dtw(x[[c],:],y[[c],:],bounding_matrix)
    return partial_dist

@njit(cache=True)
def _cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
) -> np.ndarray:
    """Dtw distance compiled to no_python.

    Series should be shape (d, m), where d is the number of dimensions, m the series
    length. Series can be different lengths.

    Parameters
    ----------
    x: np.ndarray (2d array of shape dxm1).
        First time series.
    y: np.ndarray (2d array of shape dxm1).
        Second time series.
    bounding_matrix: np.ndarray (2d array of shape m1xm2)
        Bounding matrix where the index in bound finite values (0.) and indexes
        outside bound points are infinite values (non finite).

    Returns
    -------
    cost_matrix: np.ndarray (of shape (n, m) where n is the len(x) and m is len(y))
        The dtw cost matrix.
    """
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.full((x_size + 1, y_size + 1), np.inf)
    cost_matrix[0, 0] = 0.0

    for i in range(x_size):
        for j in range(y_size):
            if np.isfinite(bounding_matrix[i, j]):
                sum_cur = 0
                for k in range(dimensions):
                    sum_cur += (x[k][i] - y[k][j]) ** 2
                cost_matrix[i + 1, j + 1] = np.sqrt(sum_cur)
                cost_matrix[i + 1, j + 1] += min(
                    cost_matrix[i, j + 1], cost_matrix[i + 1, j], cost_matrix[i, j]
                )

    return cost_matrix[1:, 1:]