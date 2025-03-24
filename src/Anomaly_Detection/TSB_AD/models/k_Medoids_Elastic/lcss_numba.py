import numpy as np

from .dist import distance_matrix, distance_matrix_independent
from numba import njit, prange

def lcss_d_dist_mat(X,Y,bounding_matrix,epsilon,adaptive_scaling=False, **kwargs):
    lcss_lamb = lambda x,y: lcss(x,y,bounding_matrix,epsilon)
    return distance_matrix(X,Y,lcss_lamb,adaptive_scaling=adaptive_scaling, **kwargs)

def lcss_i_dist_mat(X,Y,bounding_matrix,epsilon,adaptive_scaling=False, **kwargs):
    lcss_lamb = lambda x,y: lcss(x,y,bounding_matrix,epsilon)
    return distance_matrix_independent(X,Y,lcss_lamb,adaptive_scaling=adaptive_scaling, **kwargs)

def lcss(x,y,bounding_matrix,epsilon):
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = _sequence_cost_matrix(x, y, bounding_matrix, epsilon)
    return 1 - float(cost_matrix[x_size, y_size] / min(x_size, y_size))

@njit(cache=True)
def _sequence_cost_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bounding_matrix: np.ndarray,
    epsilon: float,
):
    """Compute the lcss cost matrix between two time series.

    Parameters
    ----------
    x: np.ndarray (2d array), first time series.
    y: np.ndarray (2d array), second time series.
    bounding_matrix: np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Bounding matrix where the values in bound are marked by finite values and
        outside bound points are infinite values.
    epsilon : float
        Matching threshold to determine if distance between two subsequences are
        considered similar (similar if distance less than the threshold).

    Returns
    -------
    np.ndarray (2d of size mxn where m is len(x) and n is len(y))
        Lcss cost matrix between x and y.
    """
    dimensions = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    cost_matrix = np.zeros((x_size + 1, y_size + 1))
    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            if np.isfinite(bounding_matrix[i - 1, j - 1]):
                curr_dist = 0
                for k in range(dimensions):
                    curr_dist += (x[k][i - 1] - y[k][j - 1]) ** 2
                curr_dist = np.sqrt(curr_dist)
                if curr_dist <= epsilon:
                    cost_matrix[i, j] = 1 + cost_matrix[i - 1, j - 1]
                else:
                    cost_matrix[i, j] = max(
                        cost_matrix[i, j - 1], cost_matrix[i - 1, j]
                    )
    return cost_matrix