import numpy as np
from .dist import distance_matrix, distance_matrix_independent

from numba import njit,prange

def msm_d_dist_mat(X,Y,c, adaptive_scaling=False, n_jobs=-1):
    msm_lamb = lambda x,y: msm_d(x,y,c)
    return distance_matrix(X,Y,msm_lamb,adaptive_scaling=adaptive_scaling, n_jobs=n_jobs)

def msm_i_dist_mat(X,Y,c, adaptive_scaling=False, n_jobs=-1):
    msm_lamb = lambda x,y: msm_d(x,y,c)
    return distance_matrix_independent(X,Y,msm_lamb,adaptive_scaling=adaptive_scaling, n_jobs=n_jobs)

@njit(cache=True)
def _eucl(x, y):
    return np.linalg.norm(x - y)

@njit(cache=True)
def msm_d(A, B, c):
    """
    Merge-split-move distance for MTS with global alignment.
    Translated from the java code of shifaz23.
    Parameters:
    A: an mts formatted (C,T)
    B: an mts formatted (C,T)
    c: the cost of the merge-split-move operation
    """
    # Initialize cost matrix
    n = A.shape[1]
    m = B.shape[1]
    cost_matrix = np.zeros((n, m))

    # Initialize the cost matrix
    cost_matrix[0, 0] = _eucl(A[:, 0], B[:, 0])
    for i in range(1, n):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + _costForVectorsWithEnvelop(A, i, i - 1, B, 0, c)
    for j in range(1, m):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + _costForVectorsWithEnvelop(B, j, j - 1, A, 0, c)

    # Main loop
    for i in range(1, n):
        for j in range(1, m):
            diagonal = cost_matrix[i - 1, j - 1] + _eucl(A[:, i], B[:, j])
            left = cost_matrix[i - 1, j] + _costForVectorsWithEnvelop(A, i, i - 1, B, j, c)
            up = cost_matrix[i, j - 1] + _costForVectorsWithEnvelop(B, j, j - 1, A, i, c)
            cost_matrix[i, j] = min(diagonal, left, up)

    # Return the distance
    return cost_matrix[-1, -1]

@njit(cache=True)
def _costForVectorsWithEnvelop(X: np.ndarray, xi: int, xi_prev: int, Y: np.ndarray, yi: int, c: float):
    """
    cost_multiv algorithm of Shifaz et al. 2023.
    Computing the cost for multivariate data, using the hypersphere approach.
    Parameters:
    X: an mts formatted (C,T)
    xi: the current index of the first MTS
    xi_prev: the previous index of the first MTS
    Y: an mts formatted (C,T)
    yj: the current index of the second MTS
    c: the cost of the merge-split-move operation
    """

    # Get the radius of the hypersphere
    radius = _eucl(X[:, xi_prev], Y[:, yi]) / 2

    # Get the center of the hypersphere
    center = (X[:, xi_prev] + Y[:, yi]) / 2

    # Get the distance of Xi to the center
    dist_xi = _eucl(X[:, xi], center)

    # Determine the cost
    if (dist_xi <= radius):
        return c
    else:
        dist_to_x_prev = _eucl(X[:, xi], X[:, xi_prev])
        dist_to_y = _eucl(X[:, xi], Y[:, yi])

        if (dist_to_x_prev < dist_to_y):
            return c + dist_to_x_prev
        else:
            return c + dist_to_y