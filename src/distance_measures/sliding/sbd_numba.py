from numpy.linalg import norm
from numpy.fft import fft, ifft
import numpy as np
from numpy.linalg import norm, eigh
from numpy.fft import fft2, ifft2

from numba import prange,jit,njit,objmode

# @njit
def SBD_global_ncc_c_3dim(x, y,x_fft,y_fft,adaptive_scaling=False):

    # Get length of each axis
    x_len_axis0 = int(x.shape[0])
    x_len_axis1 = int(x.shape[1])

    # Make them become the next power of two
    fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()
    fft_size_axis1 = 1 << (2*x_len_axis1-1).bit_length()

    # Cross correlation
    if adaptive_scaling:
        a = np.sum(np.multiply(x,y),axis=0,keepdims=True) / np.sum(np.multiply(y,y),axis=0,keepdims=True)
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

@njit(parallel=True)
def SBD_Global_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling=False):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            x_fft = X_ffts[i]
            y_fft = Y_ffts[j]
            with objmode(ncc="float64[:]"):
                ncc = SBD_global_ncc_c_3dim(x.T,y.T,x_fft,y_fft,adaptive_scaling)
            dist = 1-ncc.max()
            
            dist_mat[i][j] = dist
    return dist_mat
#OK

def SBD_local_ncc_c_3dim(x, y,x_fft,y_fft,adaptive_scaling=False):

    x_len = x.shape[0]
    fft_size = 1 << (2*x_len-1).bit_length()

    if adaptive_scaling:
        a = np.sum(np.multiply(x,y),axis=0,keepdims=True) / np.sum(np.multiply(y,y),axis=0,keepdims=True)
        cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(a*y, fft_size, axis=0)), axis=0)
        den = norm(x, axis=(0, 1)) * norm(a*y, axis=(0, 1))
    else:
        cc = ifft(x_fft * np.conj(y_fft), axis=0)
        den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))
    if den < 1e-9:
        den = np.inf
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den
# OK

@njit(parallel=True)
def SBD_Local_distmat(X,Y,X_ffts,Y_ffts,adaptive_scaling=False):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            x_fft = X_ffts[i]
            y_fft = Y_ffts[j]
            dist = 0
            for c in range(X.shape[1]):
                with objmode(ncc="float64[:]"):
                    ncc = SBD_local_ncc_c_3dim(np.expand_dims(x[c, :], axis=1),np.expand_dims(y[c, :], axis=1),np.expand_dims(x_fft[c,:],axis=1),np.expand_dims(y_fft[c,:],axis=1),adaptive_scaling)
                dist = dist + (1 - ncc.max())
            
            dist_mat[i][j] = dist
    return dist_mat
# OK