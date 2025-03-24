import numpy as np

import math

from numpy.fft import fft, ifft
from numpy.fft import fft2, ifft2
from numba import njit,prange

from distance_measures.sliding.sbd_numba_rocket_test import SBD_global_ncc_c_3dim, NCCc

# @njit
def nextpow2(N):
    n = 1
    while n < N: n *= 2
    return n

# @njit
def PreservedEnergy(x, e):
    FFTx = fft(x, 2**np.ceil(np.log2(abs(2*len(x) - 1)))) 
    NormCumSum = np.cumsum(np.abs(FFTx**2))/np.sum(np.abs(FFTx**2))
    k = np.argwhere(NormCumSum >= (e /2))[0]
    FFTx[k+1:len(FFTx)-k-1] = 0 
    return FFTx

# @njit
def PreservedEnergy2D(x,e):

    # Get length of each axis
    x_len_axis0 = int(x.shape[0])
    x_len_axis1 = int(x.shape[1])

    # Make them become the next power of two
    fft_size_axis0 = 1 << (2*x_len_axis0-1).bit_length()
    fft_size_axis1 = 1 << (2*x_len_axis1-1).bit_length()

    FFTx = fft2(x,(fft_size_axis0,fft_size_axis1))
    NormCumSum = np.cumsum(np.abs(FFTx**2))/np.sum(np.abs(FFTx**2))

    k = np.argwhere(NormCumSum >= (e /2))[0]
    FFTx[k+1:len(FFTx)-k-1] = 0 

    return FFTx

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

    fft_size_axis0 = nextpow2(x_len_axis0)
    fft_size_axis1 = nextpow2(x_len_axis1)

    FFTx = fft2(x,(fft_size_axis0,fft_size_axis1))
    FFTy = fft2(y,(fft_size_axis0,fft_size_axis1))

    cc = ifft2(FFTx*np.conj(FFTy))

    y_norm = np.linalg.norm(y)
    x_norm = np.linalg.norm(x)
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

# @njit
def NCC(x, y, e):
    FFTx = PreservedEnergy(x, e)
    FFTy = PreservedEnergy(y, e)
    return ifft(FFTx * FFTy)/np.dot(np.linalg.norm(x),np.linalg.norm(y))

# @njit
def SumExpNCC(x, y, gamma, e):
    ncc = NCCc(x, y)
    return np.sum(np.exp(gamma * ncc))
# @njit
def SumExpNCC2D(x,y,gamma,e):
    ncc2 = NCC2D(x,y,e)
    return np.sum(np.exp(gamma * ncc2))

# @njit
def SINK(x, y, gamma=0.1, e=math.e):
    r"""
    Shift Invariant Kernel (SINK) [1]_ [2]_
    computes the distance between time series X and Y by summing all weighted elements of the Coefficient Normalized Cross-Correlation 
    (:math:`NCC_c`) sequence between :math:`X` and :math:`Y`. 
    Formally, SINK is defined as follows:

    .. math::

        \begin{equation}
            SINK(x,y,\gamma) = \sum_{i=1}^ne^{\gamma * NCCc_i(x,y)}
        \end{equation} 
    
    where :math:`\gamma > 0`.

    :param x: time series :code:`x`
    :type x: np.array
    :param y: time series :code:`x`
    :type y: np.array
    :param gamma: bandwidth paramater that determines weights for each inner product through :math:`k'(\vec{x}, \vec{y}, \gamma) = e^{\gamma<\vec{x}, \vec{y}>}`
    :type: float, :math:`\gamma` > 0
    :param e: constant, default to :math:`e`
    :return: the SINK distance

    **References**

    .. [1] John Paparrizos and Michael Franklin. “GRAIL: Efficient Time-SeriesRepresentation Learning”. In:Proceedings of the VLDB Endowment12(2019)

    .. [2] Amaia Abanda, Usue Mor, and Jose A. Lozano. “A review on distancebased time series classification”. In:Data Mining and Knowledge Discovery12.378–412 (2019)
    
    """

    return SumExpNCC(x, y, gamma, e) 
    # return SumExpNCC(x, y, gamma, e) / math.sqrt(SumExpNCC(x, x, gamma, e)*SumExpNCC(y, y, gamma, e)) 

# @njit
def SINK2D(x,y,gamma=0.01,e=math.e):
    return SumExpNCC2D(x, y, gamma, e)
    # return SumExpNCC2D(x, y, gamma, e) / math.sqrt(SumExpNCC2D(x, x, gamma, e)*SumExpNCC2D(y, y, gamma, e)) 

# @njit
def SINK_i(x,y,gamma=0.01,e=math.e):
    dist = 0
    for channel in range(x.shape[0]):
        dist += SINK(x[channel],y[channel],gamma,e)
    return dist

# @njit
def SINK_d(x,y,gamma=0.01,e=math.e):
    dist = SINK2D(x,y,gamma,e)
    return dist

# @njit
def sink_i_all_numba(X,Y,gamma=0.01,e=math.e):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dist_mat[i][j] = SINK_i(X[i],Y[j],gamma,e)
    return dist_mat

# @njit
def sink_d_all_numba(X,Y,gamma=0.01,e=math.e):
    dist_mat = np.zeros((X.shape[0],Y.shape[0]))

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dist_mat[i][j] = SINK_d(X[i],Y[j],gamma,e)

    return dist_mat
