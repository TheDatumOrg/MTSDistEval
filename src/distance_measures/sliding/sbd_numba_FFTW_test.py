from numba import prange,jit,njit,objmode
import numpy as np
from numpy.linalg import norm
import pyfftw

# SBD local FFTW
@njit
def pad_with_constant(array, pad_width, constant_value=0):
    # Assuming pad_width is a tuple of (before, after) padding sizes
    result = np.empty(len(array) + sum(pad_width), array.dtype)
    result[:pad_width[0]] = constant_value
    result[pad_width[0]:-pad_width[1]] = array
    result[-pad_width[1]:] = constant_value
    return result

# One dimension array
def CC_FFTW(x, y):
    x_len = x.shape[0]
    # fft_size = 1 << (2*x_len-1).bit_length()
    fft_size = np.intp(2**(np.ceil(np.log2(2*x_len-1))))
    padding_size = fft_size - x_len
    x_pad = pad_with_constant(x, (0, padding_size), constant_value=0)
    y_pad = pad_with_constant(y, (0, padding_size), constant_value=0)
    # Create FFT objects
    fft_x = pyfftw.builders.fft(x_pad)
    fft_y = pyfftw.builders.fft(y_pad)
    # Execute FFTs
    X = fft_x()
    Y = fft_y()
    # Multiply by complex conjugate of Y
    result = X * np.conj(Y)
    # Create IFFT object for the result
    ifft_result = pyfftw.builders.ifft(result)
    # Execute the IFFT
    cc = ifft_result()
    # z contains the result of the inverse FFT of the product
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return cc

def SBD_local_ncc_c_3dim(x, y, adaptive_scaling=False):
    if adaptive_scaling:
        a = (x@y) / (y@y)
        # cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(a*y, fft_size, axis=0)), axis=0)
        cc = CC_FFTW(x, a*y)
        den = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum((a*y) * (a*y)))
    else:
        cc = CC_FFTW(x, y)
        den = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if den < 1e-9:
        den = np.inf
    # print(np.real(cc))
    return np.real(cc) / den

@njit(parallel=True)
def SBD_Local_all_FFTW(X,Y, adaptive_scaling=False):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            x = X[i]
            y = Y[j]
            dist = 0
            for c in range(X.shape[1]):
                with objmode(ncc="float64[:]"):
                    ncc = SBD_local_ncc_c_3dim(x[c, :], y[c, :], adaptive_scaling)
                dist = dist + (1 - ncc.max())
            dist_mat[i][j] = dist
    return dist_mat
# OK