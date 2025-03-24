import numpy as np
import math
from numpy.fft import fft, ifft
from numpy.fft import fft2, ifft2
from src.distance_measures.dist import distance_matrix, distance_matrix_independent
from numba import njit,prange
from .ncc import NCCc, NCC2D

# @njit
def SumExpNCC(x, y, gamma, e):
    ncc = NCCc(x, y)
    return np.sum(np.exp(gamma * ncc))

# @njit
def SumExpNCC2D(x,y,gamma,e):
    ncc2 = NCC2D(x,y,e)
    return np.sum(np.exp(gamma * ncc2))

def SINK(x, y, gamma=0.1, e=math.e):
    return SumExpNCC(x, y, gamma, e) 

def SINK_denom(x, y, gamma=0.1, e=math.e):
    return SumExpNCC(x, y, gamma, e) / math.sqrt(SumExpNCC(x, x, gamma, e)*SumExpNCC(y, y, gamma, e)) 

def SINK2D(x,y,gamma=0.01,e=math.e):
    return SumExpNCC2D(x, y, gamma, e)

def SINK2D_denom(x,y,gamma=0.01,e=math.e):
    return SumExpNCC2D(x, y, gamma, e) / math.sqrt(SumExpNCC2D(x, x, gamma, e)*SumExpNCC2D(y, y, gamma, e)) 

def sink_d_all(X,Y,adaptive_scaling,**kwargs):
    return distance_matrix(X,Y,SINK2D,adaptive_scaling=adaptive_scaling,**kwargs)

def sink_d_denom_all(X,Y,adaptive_scaling,**kwargs):
    return distance_matrix(X,Y,SINK2D_denom,adaptive_scaling=adaptive_scaling,**kwargs)

def sink_i_all(X,Y,adaptive_scaling,**kwargs):
    return distance_matrix_independent(X,Y,SINK,adaptive_scaling=adaptive_scaling,**kwargs)

def sink_i_denom_all(X,Y,adaptive_scaling,**kwargs):
    return distance_matrix_independent(X,Y,SINK_denom,adaptive_scaling=adaptive_scaling,**kwargs)
