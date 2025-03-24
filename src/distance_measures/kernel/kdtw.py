import numpy as np

from src.distance_measures.dist import distance_matrix, distance_matrix_independent

from src.distance_measures.kernel.KDTW.kdtw_cdist import kdtw

def kdtw_wrapper(x,y,sigma=1.0,epsilon=1e-3):
    dist = kdtw(x,y,sigma,epsilon)
    return dist

def kdtw_d_all(X,Y,adaptive_scaling,**kwargs):
    return distance_matrix(X,Y,kdtw,adaptive_scaling=adaptive_scaling,**kwargs)

def kdtw_i_all(X,Y,adaptive_scaling,**kwargs):
    return distance_matrix_independent(X,Y,kdtw,adaptive_scaling=adaptive_scaling,**kwargs)
    