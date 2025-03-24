import numpy as np
from tslearn.metrics import dtw as lcss_tslearn
from src.distance_measures.dist import local_alignment

from src.distance_measures.elastic.lower_bounding import resolve_bounding_matrix

from src.distance_measures.elastic.lcss_numba import lcss_d_dist_mat,lcss_i_dist_mat


def lcss_all(X,Y,mode='dependent',epsilon: float = 1.0,sakoe_chiba_radius: float = None,itakura_max_slope: float = None, adaptive_scaling: bool = False, **kwargs):
    if mode == 'dependent':
        # Test for sqrt(D) * epsilon
        epsilon = np.sqrt(X.shape[1]) * epsilon        
        bounding_matrix = resolve_bounding_matrix(X[0],Y[0],window=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope)
        dist_mat = lcss_d_dist_mat(X,Y,bounding_matrix,epsilon,adaptive_scaling, **kwargs)
    elif mode =='independent':
        bounding_matrix = resolve_bounding_matrix(X[0,[0],:],Y[0,[0],:],window = sakoe_chiba_radius,itakura_max_slope = itakura_max_slope)
        dist_mat = lcss_i_dist_mat(X,Y,bounding_matrix,epsilon,adaptive_scaling, **kwargs)

    return dist_mat

# # Longest common subsequence LCSS
# # A, B is (C, T) or (T,)
# def lcss(A,B):
#     assert len(A.shape) == len(B.shape)

#     m = min(A.shape[-1], B.shape[-1])

#     # Epsilon = sigma A / 2
#     eps = np.std(A) / 2
#     delta = m / 20

#     return 1 - lcss_tslearn(A.T,B.T, eps=eps, sakoe_chiba_radius=delta)

# # LCSS-Dependent
# # A, B: (C, T)
# def lcss_d(A,B):
#     '''Align the first dimensions. From Jens. Since our data has the same dimension, I comment it and feel free to discuss.
#     A,B = align_first_dimensions(A,B)'''
#     return lcss(A, B)

# # LCSS-Independent
# # A, B: (C, T)
# def lcss_i(A,B):
#     return local_alignment(A, B, lcss)