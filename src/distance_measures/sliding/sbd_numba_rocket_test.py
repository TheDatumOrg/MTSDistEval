from numpy.linalg import norm
from numpy.fft import fft, ifft
import numpy as np
from numpy.linalg import norm, eigh
from numpy.fft import fft2, ifft2
from numba import prange,jit,njit,objmode
from ..kernel.ncc import NCCc

# SBD-Local OK
def SBD_local_ncc_c_3dim(x,y,x_fft,y_fft,adaptive_scaling=False):
    x_len = x.shape[0]
    fft_size = np.intp(2**(np.ceil(np.log2(2*x_len-1))))

    if adaptive_scaling:
        a = (x.T @ y) / (y.T @ y)
        cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(a*y, fft_size, axis=0)), axis=0)
        den = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum((a*y) * (a*y)))
    else:
        cc = ifft(x_fft * np.conj(y_fft), axis=0)
        den = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if den < 1e-9:
        den = np.inf
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den


# @njit
def SBD_Local_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling=False, **kwargs):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            x_fft = X_ffts[i]
            y_fft = Y_ffts[j]
            dist = 0
            for c in range(X.shape[1]):
                ncc = SBD_local_ncc_c_3dim(np.expand_dims(x[c, :], axis=1),np.expand_dims(y[c, :], axis=1),np.expand_dims(x_fft[c,:],axis=1),np.expand_dims(y_fft[c,:],axis=1),adaptive_scaling)
                dist = dist + (1 - ncc.max())
            
            dist_mat[i][j] = dist
    return dist_mat

"""
  Generic distance matrix computation
  X: an mts dataset formatted (N,C,T) Test
  Y: an mts dataset formatted (M,C,T) Train
  Return is a (N, M) distance matrix
"""
# @njit
def SBD_Local_all_rocket(X,Y,adaptive_scaling=False, **kwargs):
    x_len_axis0 = int(X.shape[2])

    # fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()

    fft_size_axis0 = np.intp(2**(np.ceil(np.log2(2*x_len_axis0-1))))
    
    X_ffts = np.fft.fft(X,fft_size_axis0,axis=2)
    Y_ffts = np.fft.fft(Y,fft_size_axis0,axis=2)

    dist_mat = SBD_Local_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling, **kwargs)

    return dist_mat
# OK

# SBD-global
# @njit
def SBD_global_ncc_c_3dim(x, y,x_fft,y_fft,adaptive_scaling=False):

    # Get length of each axis
    x_len_axis0 = int(x.shape[0])
    x_len_axis1 = int(x.shape[1])

    # Make them become the next power of two
    # fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()
    # fft_size_axis1 = 1 << (2*x_len_axis1-1).bit_length()
    fft_size_axis0 = np.intp(2**(np.ceil(np.log2(2*x_len_axis0-1))))
    fft_size_axis1 = np.intp(2**(np.ceil(np.log2(2*x_len_axis1-1))))

    # Cross correlation
    if adaptive_scaling:
        # a = np.sum(np.multiply(x,y),axis=0,keepdims=True) / np.sum(np.multiply(y,y),axis=0,keepdims=True)
        a = np.sum(np.multiply(x,y),axis=0) / np.sum(np.multiply(y,y),axis=0)
        cc = ifft2(fft2(x, (fft_size_axis0, fft_size_axis1)) * np.conj(fft2(a*y, (fft_size_axis0, fft_size_axis1))))
        y_norm = norm(a*y)
        x_norm = norm(x)
    else:
        cc = ifft2(x_fft * np.conj(y_fft))
        y_norm = norm(y)
        x_norm = norm(x)
    den = x_norm * y_norm
    if den < 1e-9:
        den = np.inf
    # Reordering
    # Shape: 2 * x_len_axis0 - 1 ; 2 * x_len_axis1 - 1
    cc = np.concatenate((cc[-(x_len_axis0-1):, :], cc[:x_len_axis0,:]), axis=0)
    cc = np.concatenate((cc[:,-(x_len_axis1-1):], cc[:, :x_len_axis1]), axis=1)

    real_cc = np.real(cc)
    real_cc_middle = real_cc[:, x_len_axis1-1]
    ncc =  real_cc_middle / den

    return ncc
# OK

# @njit
def SBD_Global_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling=False, **kwargs):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            x_fft = X_ffts[i]
            y_fft = Y_ffts[j]
            ncc = SBD_global_ncc_c_3dim(x.T,y.T,x_fft,y_fft,adaptive_scaling)
            dist = 1-ncc.max()
            
            dist_mat[i][j] = dist
    return dist_mat

# @njit
def SBD_Global_all_rocket(X,Y,adaptive_scaling=False, **kwargs):

    # Get length of each axis
    x_len_axis0 = int(X.shape[2])
    x_len_axis1 = int(X.shape[1])

    # Make them become the next power of two
    # fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()
    fft_size_axis0 = np.intp(2**(np.ceil(np.log2(2*x_len_axis0-1))))
    
    # fft_size_axis1 = 1 << (2*x_len_axis1-1).bit_length()
    fft_size_axis1 = np.intp(2**(np.ceil(np.log2(2*x_len_axis1-1))))

    X_ffts = np.fft.fft2(np.transpose(X,axes=(0,2,1)), (fft_size_axis0, fft_size_axis1)) 
    Y_ffts = np.fft.fft2(np.transpose(Y,axes=(0,2,1)), (fft_size_axis0, fft_size_axis1)) 

    dist_mat = SBD_Global_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling, **kwargs)

    return dist_mat
# OK

# t1, t2: 2-dimensions (C, T)
# @njit
def sbd_dep_multi_rocket(T1,T2,adaptive_scaling=False):
    # Transpose to (T, C)
    T1 = T1.T
    T2 = T2.T
    d = T1.shape[1]
    cc_ = np.zeros((2*T1.shape[0]-1,), dtype=np.float64)
    for d_i in range(d):
        if adaptive_scaling:
            a = np.array(T1[:,d_i] @ T2[:,d_i]) / np.array(T2[:,d_i] @ T2[:,d_i])
            cc_ = cc_ + NCCc(T1[:,d_i],a*T2[:,d_i])    
        else:
            cc_ = cc_ + NCCc(T1[:,d_i],T2[:,d_i])

    maxCC = np.max(cc_)
    dist = d - maxCC
    return dist

# @njit
def m_sbd_all_rocket(X,Y,adaptive_scaling=False):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            dist = sbd_dep_multi_rocket(x,y,adaptive_scaling)
            dist_mat[i][j] = dist
    return dist_mat
# OK