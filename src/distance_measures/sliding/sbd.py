import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft
import numpy as np
from numpy.linalg import norm, eigh
from numpy.fft import fft2, ifft2

from distance_measures.sliding.sbd_numba import SBD_Global_distmat,SBD_Local_distmat
from distance_measures.dist import distance_matrix

# m-SBD Intialially written by https://github.com/mertozer/mts-clustering/tree/master, modifed by Haojun Li
# t1, t2: 1-dimension (T,)
def NCCc(t1,t2):
	len_ = len(t1)
	fftLen = int(2**np.ceil(np.log2(abs(2*len_ - 1))))
	
	r = ifft(fft(t1, fftLen) * np.conj(fft(t2, fftLen)))
	r = np.concatenate((r[-(len_-1):], r[:len_]))
	return np.real(r)/((norm(t1) * norm(t2)) + np.finfo(float).eps)

# t1, t2: 2-dimensions (C, T)
def sbd_dep_multi(T1,T2,**kwargs):

	# Transpose to (T, C)
	T1 = T1.T
	T2 = T2.T
	
	d = T1.shape[1]
	
	cc_ = 0
	for d_i in range(d):
		cc_ += NCCc(T1[:,d_i],T2[:,d_i])
 
    # Updated by Haojun: np.max(cc_) does not use the feature of shifting window. 
	maxCC = np.max(cc_)
	dist = d - maxCC
	
	return dist
# m-SBD OK

# SBD-Global
# x, y: (T, C)
def SBD_global_ncc_c_3dim(x, y):
    # Denominator
    y_norm = norm(y)
    x_norm = norm(x)
    den = x_norm * y_norm
    if den < 1e-9:
        den = np.inf

    # Get length of each axis
    x_len_axis0 = x.shape[0]
    x_len_axis1 = x.shape[1]

    # Make them become the next power of two
    fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()
    fft_size_axis1 = 1 << (2*x_len_axis1-1).bit_length()

    # Cross correlation
    cc = ifft2(fft2(x, (fft_size_axis0, fft_size_axis1)) * np.conj(fft2(y, (fft_size_axis0, fft_size_axis1))))
    
    # Reordering
    # Shape: 2 * x_len_axis0 - 1 ; 2 * x_len_axis1 - 1
    cc = np.concatenate((cc[-(x_len_axis0-1):, :], cc[:x_len_axis0,:]), axis=0)
    cc = np.concatenate((cc[:,-(x_len_axis1-1):], cc[:, :x_len_axis1]), axis=1)

    real_cc = np.real(cc)
    real_cc_middle = real_cc[:, x_len_axis1-1]
    ncc =  real_cc_middle / den
    return ncc


def SBD_global_dist(A,B):
    return 1 - SBD_global_ncc_c_3dim(A.T, B.T).max()
# OK

'''
X: (N, C, T)
Y: (M, C, T)
'''
# def SBD_Global_all(X, Y,adaptive_scaling=False):
#   # dist = np.zeros((X.shape[0], Y.shape[0]))
#   # for n in range(0, X.shape[0]):
#   #   for m in range(0, Y.shape[0]):
#   #     dist[n, m] = 1 - SBD_global_ncc_c_3dim(X[n].T, Y[m].T).max()
#   # return dist
#   return distance_matrix(X, Y, SBD_global_dist,adaptive_scaling=adaptive_scaling)
# SBD-Global O

def SBD_Global_all(X,Y,adaptive_scaling=False):

    # Get length of each axis
    x_len_axis0 = int(X.shape[2])
    x_len_axis1 = int(X.shape[1])

    # Make them become the next power of two
    fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()
    fft_size_axis1 = 1 << (2*x_len_axis1-1).bit_length()

    X_ffts = np.fft.fft2(np.transpose(X,axes=(0,2,1)), (fft_size_axis0, fft_size_axis1)) 
    Y_ffts = np.fft.fft2(np.transpose(Y,axes=(0,2,1)), (fft_size_axis0, fft_size_axis1)) 

    dist_mat = SBD_Global_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling)

    return dist_mat
#OK

# def SBD_Global_faster(X,Y,adaptive_scaling=False,n_jobs=-1):
#   pass
# SBD-Local
# x, y: (T, 1)
def SBD_local_ncc_c_3dim(x, y):
    den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))

    if den < 1e-9:
        den = np.inf

    x_len = x.shape[0]
    fft_size = 1 << (2*x_len-1).bit_length()

    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den

"""
  Generic distance matrix computation
  X: an mts dataset formatted (N,C,T) Test
  Y: an mts dataset formatted (M,C,T) Train
  Return is a (N, M) distance matrix
"""
# def SBD_Local_all(X, Y,adaptive_scaling=False):
#   # dist = np.zeros((X.shape[0], Y.shape[0]))
#   # for n in range(0, X.shape[0]):
#   #   for m in range(0, Y.shape[0]):
#   #     d_sum = 0
#   #     for d in range(0, X.shape[1]):
#   #       d_sum = d_sum + 1 - SBD_local_ncc_c_3dim(np.expand_dims(X[n, d, :], axis=1), np.expand_dims(Y[m, d, :], axis=1)).max()
                
#   #     dist[n, m] = d_sum
#   # return dist
#   return distance_matrix(X,Y,SBD_local,adaptive_scaling=adaptive_scaling)

def SBD_Local_all(X,Y,adaptive_scaling=False):
    x_len_axis0 = int(X.shape[2])

    fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()

    X_ffts = np.fft.fft(X,fft_size_axis0,axis=2)
    Y_ffts = np.fft.fft(Y,fft_size_axis0,axis=2)

    dist_mat = SBD_Local_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling)

    return dist_mat
# OK


def SBD_local(A,B):
    d_sum=0
    for d in range(0, A.shape[0]):
      d_sum = d_sum + 1 - SBD_local_ncc_c_3dim(np.expand_dims(A[d, :], axis=1), np.expand_dims(B[d, :], axis=1)).max()
    return d_sum
  
# SBD-Local OK

# ''' X: an mts formatted (N,C,T)
#     Y: an mts formatted (M,C,T)
#     Return is a (N, M) distance matrix'''
# def sbd_all(X, Y, shift):
#   dist = np.zeros((X.shape[0], Y.shape[0]))
#   for n in range(0, X.shape[0]):
#     for m in range(0, Y.shape[0]):
#       dist[n, m], optShift, opt_t2 = sbd_dep_multi(Y[m].T, X[n].T, shift)
#   return dist
# m-SBD OK