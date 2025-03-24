import numpy as np
from .dist import local_alignment
from sktime.distances import msm_distance

from .msm_numba import msm_d_dist_mat,msm_i_dist_mat

def msm_all(X,Y,mode='dependent',c=1, adaptive_scaling=False, **kwargs):
    if mode =='dependent':
        # c * sqrt(D)
        c = c * np.sqrt(X.shape[1])
        dist_mat = msm_d_dist_mat(X,Y,c, adaptive_scaling, **kwargs)
    elif mode == 'independent':
        dist_mat = msm_i_dist_mat(X,Y,c, adaptive_scaling, **kwargs)
    return dist_mat

   