import numpy as np
from .lower_bounding import resolve_bounding_matrix

from .erp_numba import erp_d_dist_mat,erp_i_dist_mat

def erp_all(X,Y,mode='dependent',sakoe_chiba_radius: float = None,itakura_max_slope: float = None,g=0.0, adaptive_scaling=False, **kwargs):
    if mode == 'dependent':
        bounding_matrix = resolve_bounding_matrix(X[0],Y[0],window=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope)
        dist_mat = erp_d_dist_mat(X,Y,bounding_matrix,g,adaptive_scaling, **kwargs)
    elif mode =='independent':
        bounding_matrix = resolve_bounding_matrix(X[0,[0],:],Y[0,[0],:],window = sakoe_chiba_radius,itakura_max_slope = itakura_max_slope)
        dist_mat = erp_i_dist_mat(X,Y,bounding_matrix,g,adaptive_scaling, **kwargs)

    return dist_mat