import numpy as np
from numba import njit,prange

@njit(parallel=True,cache=True)
def frob_all(xDists, yDists):
    """
    Compute frob distance matrix between two MTS datasets.

    Parameters
    ----------
    xDists : np.ndarray, shape=(n_mts, n_dimensions, n_dimensions)
        The internal distance matrices for X.
    yDists : np.ndarray, shape=(n_mts, n_dimensions, n_dimensions)
        The internal distance matrices for Y.

    Returns
    -------
    np.ndarray, shape=(n_mts, n_mts)
        The distance matrix between X and Y.
    """
    xn = xDists.shape[0]
    yn = yDists.shape[0]

    dist_mat = np.zeros((xn, yn))
    for i in prange(xn):
        for j in range(yn):
            dist_mat[i, j] = np.square(xDists[i] - yDists[j]).sum()

    return dist_mat

@njit(cache=True)
def lp(X,Y,p):
    """
    Compute the L2 distance between two MTS.

    Parameters
    ----------
    X : np.ndarray, shape=(n_dimensions, n_samples)
        The first MTS.
    Y : np.ndarray, shape=(n_dimensions, n_samples)
        The second MTS.

    Returns
    -------
    float
        The L2 distance between X and Y.
    """
    return np.linalg.norm(X - Y, ord=p)

@njit(parallel=True,cache=True)
def frob_l2_all(X,Y):
    xn, xc, _ = X.shape
    yn, yc, _ = Y.shape

    xDists = np.zeros((xn, xc, xc))
    yDists = np.zeros((yn, yc, yc))

    # Compute the internal distance matrices for X
    for i in prange(xn):
        for c in range(xc):
            for d in range(c, xc):
                dist = lp(X[i, c], X[i, d], 2)
                xDists[i, c, d] = dist
                xDists[i, d, c] = dist

    # Compute the internal distance matrices for Y
    for i in prange(yn):
        for c in range(yc):
            for d in range(c, yc):
                dist = lp(Y[i, c], Y[i, d], 2)
                yDists[i, c, d] = dist
                yDists[i, d, c] = dist

    return frob_all(xDists, yDists)