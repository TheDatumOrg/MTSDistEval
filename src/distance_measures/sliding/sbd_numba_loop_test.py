import numpy as np
import numpy as np
from numba import prange,jit,njit,objmode

# SBD-Local
@njit
def CC_onedim(x, y):
  """Compute the cross-correlation of two 1D arrays with zero padding."""
  len_x, len_y = len(x), len(y)
  max_lag = len_x + len_y - 1  # total lags
  corr = np.zeros(max_lag)  # to store correlation for different lags
  for lag in range(max_lag):
      sum_product = 0.0
      for i in range(len_x):
          j = i + lag - len_y + 1  # corresponding index in y with zero padding
          if j >= 0 and j < len_y:
              sum_product += x[i] * y[j]
      corr[lag] = sum_product
  return corr

# Since We are using for loop, we do not need to compute fft_size
@njit
def SBD_local_ncc_c_3dim(x, y,adaptive_scaling=False):
    if adaptive_scaling:
        a = (x@y) / (y@y)
        cc = CC_onedim(x, a*y)
        den = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum((a*y) * (a*y)))
    else:
        cc = CC_onedim(x, y)        
        den = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if den < 1e-9:
        den = np.inf
    # cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)

    return cc / den
    

@njit(parallel=True)
def SBD_Local_distmat(X,Y,adaptive_scaling=False):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            dist = 0.0
            for c in range(X.shape[1]):
                ncc = SBD_local_ncc_c_3dim(x[c, :],y[c, :],adaptive_scaling)
                dist = dist + (1 - ncc.max())
            dist_mat[i][j] = dist
    return dist_mat

"""
  Generic distance matrix computation
  X: an mts dataset formatted (N,C,T) Test
  Y: an mts dataset formatted (M,C,T) Train
  Return is a (N, M) distance matrix
"""
def SBD_Local_all_loop(X,Y,adaptive_scaling=False):
  dist_mat = SBD_Local_distmat(X,Y,adaptive_scaling)
  return dist_mat
# OK