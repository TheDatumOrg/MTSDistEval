import numpy as np

from enum import Enum
from typing import Any, Callable, Union

from src.distance_measures.elastic.lower_bounding import resolve_bounding_matrix

# Dynamic Time Warping (DTW) with fixed parameters
from src.distance_measures.elastic.dtw_numba import dtw_d_dist_mat,dtw_i_dist_mat

def dtw_all(X,Y,mode='dependent',sakoe_chiba_radius=None,itakura_max_slope=None,adaptive_scaling=False, **kwargs):
    """
    Compute the DTW distance matrix between two MTS datasets.
    Parameters:
    X : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The first MTS dataset.
    Y : np.ndarray, shape=(n_mts, n_dimensions, n_samples)
        The second MTS dataset.
    """
    if mode == 'dependent':
        bounding_matrix = resolve_bounding_matrix(X[0],Y[0],window=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope)
        dist_mat = dtw_d_dist_mat(X,Y,bounding_matrix,adaptive_scaling, **kwargs)
    elif mode =='independent':
        bounding_matrix = resolve_bounding_matrix(X[0,[0],:],Y[0,[0],:],window = sakoe_chiba_radius,itakura_max_slope = itakura_max_slope)
        dist_mat = dtw_i_dist_mat(X,Y,bounding_matrix,adaptive_scaling, **kwargs)

    return dist_mat
