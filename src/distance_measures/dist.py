import numpy as np

from src.distance_measures.pairwise_dist import pairwise_distances

def distance_matrix1d(X: np.ndarray, distance_function: callable, kwargs={}):
    """
    Generic distance matrix computation
    X: an mts dataset formatted (N,C,T) Test
    Return is a (N, M) distance matrix
    """
    if kwargs is None:
        kwargs = {}
    else: pass
    
    n = X.shape[0]
    dist = np.zeros((n,n))
    for i in range(0, n):
        for j in range(i+1, n):
            dist = pairwise_distances(X[i],X[j],distance_function, n_jobs = -1, **kwargs)
            dist[i, j] = dist
            dist[j, i] = dist
    return dist

def distance_matrix(X: np.ndarray, Y: np.ndarray, distance_function: callable, adaptive_scaling=False,n_jobs=-1, **kwargs):
    """
    Generic distance matrix computation
    X: an mts dataset formatted (N,C,T) Test
    Y: an mts dataset formatted (M,C,T) Train
    Return is a (N, M) distance matrix
    """
    # dist = np.zeros((X.shape[0], Y.shape[0]))
    # for n in range(0, X.shape[0]):
    #     for m in range(0, Y.shape[0]):
    #         dist_partial = distance_function(X[n], Y[m], **kwargs)
    #         dist[n, m] = dist_partial
    dist = pairwise_distances(X,Y,distance_function,adaptive_scaling, n_jobs = n_jobs, **kwargs)
    return dist

def distance_matrix_independent(X: np.ndarray, Y: np.ndarray, distance_function: callable, adaptive_scaling=False, n_jobs=-1, **kwargs):
    """
    Generic distance matrix computation
    X: an mts dataset formatted (N,C,T) Test
    Y: an mts dataset formatted (M,C,T) Train
    Return is a (N, M) distance matrix
    """
    dist = np.zeros((X.shape[0], Y.shape[0]))
    nc = X.shape[1]
    for i in range(nc):
        # print("Computing distance for dimension " + str(i))
        Xc = X[:,[i],:]
        Yc = Y[:,[i],:]
        dist += pairwise_distances(Xc,Yc,distance_function,adaptive_scaling=adaptive_scaling, n_jobs=n_jobs, **kwargs)
    return dist

"""
Generic multivariate extension of a UTS measure through local alignment
X: an MTS with shape (C,T)
Y: an MTS with shape (C,T)
kwargs: additional parameters for the distance function
Return is a distance between X and Y
"""
def local_alignment(X: np.ndarray, Y: np.ndarray, distance_function: callable, kwargs):
    # Check if same number of dimensions
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of dimensions of the two MTS must be the same")

    dist = np.zeros((X.shape[0]))
    for c in range(0,X.shape[0]):
        dist_c = distance_function(X[c], Y[c], kwargs)
        dist[c] = dist_c

    return np.sum(dist)