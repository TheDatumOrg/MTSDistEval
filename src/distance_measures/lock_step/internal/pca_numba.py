import numpy as np
from numba import prange,njit
import math

# @njit(cache=True)
# def pca_sim(pca1, pca2):
#     return np.sum(np.square(pca1 @ pca2.T))

# Add place_holder to make it run without issue because distance_matrix passes three arguments into this function.
@njit(cache=True)
def pca_dist(pca1, pca2):
    return 1 / (1 + np.sum(np.square(pca1 @ pca2.T)))

@njit(parallel=True)
def pca_dist_mat(pcasX,pcasY,exvar=0.9):
    dist_mat = np.zeros((pcasX.shape[0],pcasY.shape[0]))
    for p in prange(pcasX.shape[0]):
        for q in range(pcasY.shape[0]):
            dist_mat[p,q] = pca_dist(pcasX[p],pcasY[q])
    return dist_mat

@njit(cache=True)
def eros(pca1, pca2, w):
    """
    Compute the extended frobenius norm between the principal components of two MTS.
    The weights are based on the average eigenvalues of the covariance matrices of the whole dataset.
    """
    # Compute the weighted sum of dots of the eigenvectors
    s = 0
    for i in range(pca1.shape[0]):
        s += w[i] * np.abs(np.dot(pca1[i], pca2[i]))

    return s

@njit(cache=True)
def eros_dist(pca1, pca2, w):
    sim = min(eros(pca1, pca2, w), 1) # Set max similarity to 1
    return math.sqrt(2 - 2*sim)

@njit(cache=True)
def eros_dist_mat(pcasX,pcasY,weights):
    dist_mat = np.zeros((pcasX.shape[0],pcasY.shape[0]))
    for p in prange(pcasX.shape[0]):
        for q in range(pcasY.shape[0]):
            dist_mat[p,q] = eros_dist(pcasX[p],pcasY[q],weights)
    return dist_mat