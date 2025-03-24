from numba import prange,jit,njit,objmode
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2
from numpy.linalg import norm

@njit
def PreservedEnergy(x, e):
    FFTx = fft(x, 2**np.ceil(np.log2(abs(2*len(x) - 1)))) 
    NormCumSum = np.cumsum(np.abs(FFTx**2))/np.sum(np.abs(FFTx**2))
    k = np.argwhere(NormCumSum >= (e /2))[0]
    FFTx[k+1:len(FFTx)-k-1] = 0 
    return FFTx

@njit
def PreservedEnergy2D(x,e):
    # Get length of each axis
    x_len_axis0 = int(x.shape[0])
    x_len_axis1 = int(x.shape[1])

    # Make them become the next power of two
    fft_size_axis0 = nextpow2(2*x_len_axis0-1)
    fft_size_axis1 = nextpow2(2*x_len_axis1-1)

    FFTx = fft2(x,(fft_size_axis0,fft_size_axis1))
    NormCumSum = np.cumsum(np.abs(FFTx**2))/np.sum(np.abs(FFTx**2))

    k = np.argwhere(NormCumSum >= (e /2))[0]
    FFTx[k+1:len(FFTx)-k-1] = 0 

    return FFTx

@njit
def NCC(x, y, e):
    FFTx = PreservedEnergy(x, e)
    FFTy = PreservedEnergy(y, e)
    return ifft(FFTx * FFTy)/np.dot(np.linalg.norm(x),np.linalg.norm(y))

@njit
def nextpow2(N):
    n = 1
    while n < N: n *= 2
    return n

# @njit
def NCCc(t1,t2):
    t1 = t1.squeeze()
    t2 = t2.squeeze()

    len_ = len(t1)
    fftLen = int(2**np.ceil(np.log2(abs(2*len_ - 1))))

    r = ifft(fft(t1, fftLen) * np.conj(fft(t2, fftLen)))
    r = np.concatenate((r[-len_+1:], r[:len_]))

    return (r/((norm(t1) * norm(t2)) + 2.220446049250313e-16)).real

# @njit
def NCC2D(x,y,e):
    # FFTx = PreservedEnergy2D(x,e)
    # FFTy = PreservedEnergy2D(y,e)

    x= x.T
    y= y.T

    x_len_axis0 = int(x.shape[0])
    x_len_axis1 = int(x.shape[1])

    # Make them become the next power of two
    # fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()
    # fft_size_axis1 = 1 << (2*x_len_axis1-1).bit_length()

    fft_size_axis0 = nextpow2(2*x_len_axis0 - 1)
    fft_size_axis1 = nextpow2(2*x_len_axis1 - 1)

    FFTx = fft2(x,(fft_size_axis0,fft_size_axis1))
    FFTy = fft2(y,(fft_size_axis0,fft_size_axis1))

    cc = ifft2(FFTx*np.conj(FFTy))

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
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