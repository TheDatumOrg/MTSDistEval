import numpy as np
from src.distance_measures.lock_step.lp import lp_all,euclidean_all

''' X: an mts formatted (N,C,T)
    Y: an mts formatted (M,C,T)
    Return is a (N, M) distance matrix'''
def rbf_all(X,Y, gamma=1, adaptive_scaling=False, **kwargs):
    l2dists = euclidean_all(X,Y, adaptive_scaling=adaptive_scaling, **kwargs)
    return np.exp(-gamma * l2dists)
