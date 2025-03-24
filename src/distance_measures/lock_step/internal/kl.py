import numpy as np
import scipy.linalg as la
from scipy.linalg import LinAlgError

from distance_measures.dist import distance_matrix

# KL-divergence
# meanX, meanY: list of each dimension's mean
# covX, covY: (C, C) covariance matrix.
def _kl_mvn(meanX:np.ndarray, covX:np.ndarray, meanY:np.ndarray,covY:np.ndarray):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""

    ''' Map the means and covariances to the same subspace. From Jens, since our dimension is matching, we don't need this. free free to discuss.
    meanX, meanY = align_dimensions(meanX, meanY)
    covX, covY = align_dimensions(covX, covY)'''

    m_to, S_to = meanX, covX
    m_fr, S_fr = meanY, covY
    
    d = m_fr - m_to
    
    c, lower = la.cho_factor(S_fr)
    def solve(B):
        return la.cho_solve((c, lower), B)
    
    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.

# Parameters are the same as previous function
def _kl_mvn_sym(meanX:np.ndarray, covX:np.ndarray, meanY:np.ndarray,covY:np.ndarray):
    return (_kl_mvn(meanX, covX, meanY,covY) + _kl_mvn(meanY,covY,meanX,covX))/2.

def kl_dist(A,B):
    meanX = np.mean(A,axis=-1)
    covX = np.cov(A)

    meanY = np.mean(B,axis=-1)
    covY = np.cov(B)

    try:
        val = _kl_mvn_sym(meanX, covX, meanY, covY)
    except LinAlgError:
        val = np.nan
    return val

''' X: an mts formatted (N,C,T)
    Y: an mts formatted (M,C,T)
    Return is a (N, M) distance matrix'''
def kl_all(X, Y):
    # meansX = [np.mean(x, axis=-1) for x in X]
    # covsX = [np.cov(x) for x in X]

    # meansY = [np.mean(y, axis=-1) for y in Y]
    # covsY = [np.cov(y) for y in Y]
    
    kls = np.zeros((X.shape[0], Y.shape[0]))
    # for n in range(0, X.shape[0]):
    #     for m in range(0, Y.shape[0]):
    #         try:
    #             val = _kl_mvn_sym(meansX[n], covsX[n], meansY[m], covsY[m])
    #         except LinAlgError:
    #             val = np.nan
            
    #         kls[n, m] = val

    # return kls
    return distance_matrix(X,Y,kl_dist)
# kl.py OK