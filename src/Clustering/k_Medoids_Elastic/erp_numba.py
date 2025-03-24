# from distance_measures.elastic.dtw import resolve_bounding_matrix
from .dist import distance_matrix, distance_matrix_independent

import numpy as np
from numba import njit,prange

def erp_d_dist_mat(X,Y,bounding_matrix,g,adaptive_scaling=False, **kwargs):
    erp_lamb = lambda x,y: erp(x,y,bounding_matrix,g)
    return distance_matrix(X,Y,erp_lamb,adaptive_scaling=adaptive_scaling, **kwargs)


def erp_i_dist_mat(X,Y,bounding_matrix,g,adaptive_scaling=False, **kwargs):
    erp_lamb = lambda x,y: erp(x,y,bounding_matrix,g)
    return distance_matrix_independent(X,Y,erp_lamb,adaptive_scaling=adaptive_scaling, **kwargs)

def erp(x,y,bounding_matrix,g):
    cost_matrix = _erp_cost_matrix(x, y, bounding_matrix, g)
    return cost_matrix[-1, -1]

@njit(cache=True)
def _erp_cost_matrix(
    x: np.ndarray, y: np.ndarray, bounding_matrix: np.ndarray, g: float
):
    """Compute the erp cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (2d array)
        First time series.
    y: np.ndarray (2d array)
        Second time series.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the values in bound are marked by finite values and
        outside bound points are infinite values.
    g: float
        The reference value to penalise gaps ('gap' defined when an alignment to
        the next value (in x) in value can't be found).

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Erp cost matrix between x and y.
    """
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))
    gx_distance = np.zeros(x_size)
    gy_distance = np.zeros(y_size)
    for j in range(x_size):
        for i in range(dimensions):
            gx_distance[j] += (x[i][j] - g) * (x[i][j] - g)
        gx_distance[j] = np.sqrt(gx_distance[j])
    for j in range(y_size):
        for i in range(dimensions):
            gy_distance[j] += (y[i][j] - g) * (y[i][j] - g)
        gy_distance[j] = np.sqrt(gy_distance[j])
    # Initialization
    for j in range(1, x_size+1):
        cost_matrix[j, 0] = cost_matrix[j - 1, 0] + gx_distance[j-1]
    for j in range(1, y_size + 1):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + gy_distance[j-1]

    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                curr_dist = 0
                for k in range(dimensions):
                    curr_dist += (x[k][i - 1] - y[k][j - 1]) * (
                        x[k][i - 1] - y[k][j - 1]
                    )
                curr_dist = np.sqrt(curr_dist)
                cost_matrix[i, j] = min(
                    cost_matrix[i - 1, j - 1] + curr_dist,
                    cost_matrix[i - 1, j] + gx_distance[i - 1],
                    cost_matrix[i, j - 1] + gy_distance[j - 1],
                )
    return cost_matrix[1:, 1:]