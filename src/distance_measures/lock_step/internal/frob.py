import numpy as np
from numba import njit, prange
# Add the base path to the import path
from distance_measures.dist import distance_matrix1d, distance_matrix
from distance_measures.elastic.lower_bounding import resolve_bounding_matrix
from distance_measures.elastic.dtw_numba import dtw
from distance_measures.elastic.lcss_numba import lcss
from distance_measures.elastic.erp_numba import erp
from distance_measures.elastic.msm_numba import msm_d
from distance_measures.elastic.twe_numba import twe
from distance_measures.sliding.sbd import SBD_local_ncc_c_3dim
from sktime.distances import erp_distance

def _frob_distmat(distsX, distsY):
    return distance_matrix(distsX, distsY, lambda X,Y: np.linalg.norm(X-Y, ord='fro'))

def frob_all(X,Y, distance_function: callable):
    distsX = np.array([distance_matrix(X[i], X[i], distance_function) for i in range(X.shape[0])])
    distsY = np.array([distance_matrix(Y[i], Y[i], distance_function) for i in range(Y.shape[0])])
    return _frob_distmat(distsX, distsY)

def frobcov_all(X,Y):
    covsX = np.array([np.cov(X[i]) for i in range(X.shape[0])])
    covsY = np.array([np.cov(Y[i]) for i in range(Y.shape[0])])
    return _frob_distmat(covsX,covsY)

frob_l2_all = lambda X,Y: frob_all(X,Y, lambda x,y: np.linalg.norm(x - y, ord=2))
frob_l1_all = lambda X,Y: frob_all(X,Y, lambda x,y: np.sum(np.abs(x - y)))
frob_lz_all = lambda X,Y: frob_all(X,Y, lambda x,y: np.sum(np.log(1+np.abs(x - y))))

def _internal_bounding_matrix(X, sakoe_chiba_radius=None,itakura_max_slope=None):
    return resolve_bounding_matrix(X[0,[0],:],X[0,[0],:],window = sakoe_chiba_radius,itakura_max_slope = itakura_max_slope)

@njit(parallel=True,cache=True)
def _dtw_internals(X, bounding_matrix):
    """
    Compute the internal DTW distance matrices of one MTS dataset.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The MTS dataset

    Returns
    -------
    np.ndarray, shape=(n_mts,n_dimensions, n_dimensions)
        The internal distance matrix of X.
    """
    n,c,_ = X.shape

    dist_mat = np.zeros((n,c,c))
    for i in prange(n):
        x_exp = np.expand_dims(X[i],axis=0)
        for j in range(c):
            for k in range(j+1,c):
                dist = dtw(x_exp[:,j],x_exp[:,k],bounding_matrix)
                dist_mat[i,j,k] = dist
                dist_mat[i,k,j] = dist
    return dist_mat

def frob_dtw_all(X,Y,sakoe_chiba_radius: float = None,itakura_max_slope: float = None):
    distsX = _dtw_internals(X, _internal_bounding_matrix(X, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope))
    distsY = _dtw_internals(Y, _internal_bounding_matrix(Y, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope))

    return _frob_distmat(distsX, distsY)

@njit(parallel=True,cache=True)
def _lcss_internals(X, epsilon, bounding_matrix):
    """
    Compute the internal LCSS distance matrices of one MTS dataset.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The MTS dataset

    Returns
    -------
    np.ndarray, shape=(n_mts,n_dimensions, n_dimensions)
        The internal distance matrix of X.
    """
    n,c,_ = X.shape

    dist_mat = np.zeros((n,c,c))
    for i in prange(n):
        x_exp = np.expand_dims(X[i],axis=0)
        for j in range(c):
            for k in range(j+1,c):
                dist = lcss(x_exp[:,j],x_exp[:,k],bounding_matrix, epsilon)
                dist_mat[i,j,k] = dist
                dist_mat[i,k,j] = dist
    return dist_mat

def frob_lcss_all(X,Y,epsilon: float = 1.0, sakoe_chiba_radius: float = None,itakura_max_slope: float = None):
    distsX = _lcss_internals(X, epsilon, _internal_bounding_matrix(X, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope))
    distsY = _lcss_internals(Y, epsilon, _internal_bounding_matrix(Y, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope))

    return _frob_distmat(distsX, distsY)

@njit(parallel=True,cache=True)
def _erp_internals(X, g, bounding_matrix):
    """
    Compute the internal ERP distance matrices of one MTS dataset.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The MTS dataset

    Returns
    -------
    np.ndarray, shape=(n_mts,n_dimensions, n_dimensions)
        The internal distance matrix of X.
    """
    n,c,_ = X.shape

    dist_mat = np.zeros((n,c,c))
    for i in prange(n):
        x_exp = np.expand_dims(X[i],axis=0)
        for j in range(c):
            for k in range(j+1,c):
                dist = erp(x_exp[:,j],x_exp[:,k],bounding_matrix, g)
                dist_mat[i,j,k] = dist
                dist_mat[i,k,j] = dist
    return dist_mat

def frob_erp_all(X,Y,sakoe_chiba_radius: float = None,itakura_max_slope: float = None,g=0.0):
    distsX = _erp_internals(X, g, _internal_bounding_matrix(X, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope))
    distsY = _erp_internals(Y, g, _internal_bounding_matrix(Y, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope))

    return _frob_distmat(distsX, distsY)

@njit(parallel=True,cache=True)
def _twe_internals(X, bounding_matrix, lmbda, nu, p):
    """
    Compute the internal TWE distance matrices of one MTS dataset.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The MTS dataset

    Returns
    -------
    np.ndarray, shape=(n_mts,n_dimensions, n_dimensions)
        The internal distance matrix of X.
    """
    n,c,_ = X.shape

    dist_mat = np.zeros((n,c,c))
    for i in prange(n):
        x_exp = np.expand_dims(X[i],axis=0)
        for j in range(c):
            for k in range(j+1,c):
                dist = twe(x_exp[:,j],x_exp[:,k],bounding_matrix, lmbda, nu, p)
                dist_mat[i,j,k] = dist
                dist_mat[i,k,j] = dist
    return dist_mat

def frob_twe_all(X,Y,
                lmbda: float = 1.0,
                nu: float = .001, 
                p: float = 2.0,
                sakoe_chiba_radius: float = None,
                itakura_max_slope: float = None):
    distsX = _twe_internals(X, _internal_bounding_matrix(X, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope), lmbda, nu, p)
    distsY = _twe_internals(Y, _internal_bounding_matrix(Y, sakoe_chiba_radius=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope), lmbda, nu, p)

    return _frob_distmat(distsX, distsY)

@njit(parallel=True,cache=True)
def _msm_internals(X,c=1):
    """
    Compute the internal MSM distance matrices of one MTS dataset.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The MTS dataset

    Returns
    -------
    np.ndarray, shape=(n_mts,n_dimensions, n_dimensions)
        The internal distance matrix of X.
    """
    n,c,_ = X.shape

    dist_mat = np.zeros((n,c,c))
    for i in prange(n):
        x_exp = np.expand_dims(X[i],axis=0)
        for j in range(c):
            for k in range(j+1,c):
                dist = msm_d(x_exp[:,j],x_exp[:,k],c)
                dist_mat[i,j,k] = dist
                dist_mat[i,k,j] = dist
    return dist_mat

def frob_msm_all(X,Y,c=1):
    distsX = _msm_internals(X,c)
    distsY = _msm_internals(Y,c)
    return _frob_distmat(distsX, distsY)

def _sbd_internals(X):
    """
    Compute the internal MSM distance matrices of one MTS dataset.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The MTS dataset

    Returns
    -------
    np.ndarray, shape=(n_mts,n_dimensions, n_dimensions)
        The internal distance matrix of X.
    """
    n,c,_ = X.shape

    dist_mat = np.zeros((n,c,c))
    for i in range(n):
        for j in range(c):
            for k in range(j+1,c):
                Lt = X[i,j][:, np.newaxis]
                Rt = X[i,k][:, np.newaxis]

                sbd = 1 - SBD_local_ncc_c_3dim(Lt, Rt).max()
                dist_mat[i,j,k] = sbd
                dist_mat[i,k,j] = sbd

    return dist_mat

def frob_sbd_all(X,Y):
    distsX = _sbd_internals(X)
    distsY = _sbd_internals(Y)
    return _frob_distmat(distsX, distsY)

@njit(parallel=True,cache=True)
def _rbf_internals(X,gamma=0.1):
    """
    Compute the internal MSM distance matrices of one MTS dataset.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The MTS dataset

    Returns
    -------
    np.ndarray, shape=(n_mts,n_dimensions, n_dimensions)
        The internal distance matrix of X.
    """
    n,c,_ = X.shape

    dist_mat = np.zeros((n,c,c))
    for i in prange(n):
        for j in range(c):
            for k in range(j+1,c):
                l2 = np.linalg.norm(X[i,j] - X[i,k], ord=2)
                dist_mat[i,j,k] = l2
                dist_mat[i,k,j] = l2

    # Take rbf of the distances
    return np.exp(-gamma * dist_mat)

def frob_rbf_all(X,Y,gamma=0.1):
    distsX = _rbf_internals(X,gamma)
    distsY = _rbf_internals(Y,gamma)
    return _frob_distmat(distsX, distsY)

def frob_gak_all(X,Y,sigma=0.1):
    # TODO implement when GAK done
    pass

def frob_sink_all(X,Y,gamma,e):
    # TODO implement when SINK done
    pass
    
def frob_grail_all(X,Y):
    # TODO implement when GRAIL done
    pass