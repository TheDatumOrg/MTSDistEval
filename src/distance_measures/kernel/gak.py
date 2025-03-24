import numpy as np
import math
from scipy.spatial.distance import cdist
from tslearn.metrics import sigma_gak
from tslearn.metrics.softdtw_variants import unnormalized_gak

from src.distance_measures.dist import distance_matrix, distance_matrix_independent

import math
from numba import njit


log0 = -10000

@njit(nogil=True)
def logP(x, y):
    if x > y:
        return x + math.log1p(math.exp(y - x))
    else:
        return y + math.log1p(math.exp(x - y))
    
@njit(nogil=True)
def logGAK(x,y, sigma=1.0):
    nX, nY = x.shape[1], y.shape[1]

    # length of a column for the dynamic programming
    cl = nY + 1 

    # logM is the array that will stores two successive columns of the (nX+1) x (nY+1) table used to compute the final kernel value
    logM = np.zeros(2*cl)

    # Maximum of abs(i-j) when 1<=i<=nX and 1<=j<=nY
    trimax = nX - 1 if nX > nY else nY - 1

    logTriangularCoefficients = np.zeros(trimax + 1)

    Sig=-1/(2*sigma*sigma)

    """
    /****************************************************/
    /* First iteration : initialization of columns to 0 */
    /****************************************************/
    """
    # The left most column is all zeros..
    logM[1:cl] = log0

    # ... except for the lower-left cell which is initialized with a value of 1, i.e. a log value of 0.
    # logM[0] = 0

    # Cur and Old keep track of which column is the current one and which one is the already computed one.
    cur = 1      # Indexes [0..cl-1] are used to process the next column */
    old = 0      # Indexes [cl..2*cl-1] were used for column 0 */

    """
    /************************************************/
    /* Next iterations : processing columns 1 .. nX */
    /************************************************/
    """

    # First compute all the squared euclidean distances with numpy
    cdists = np.sum((x[..., np.newaxis] - y[..., np.newaxis, :])**2, axis=0)

    # Main loop to vary the position for i=1..nX
    for i in range(1, nX+1):
        # Special update for positions (i=1..nX,j=0)
        curpos = cur*cl  # index of the state (i,0)
        logM[curpos] = log0
        # Secondary loop to vary the position for j=1..nY
        for j in range(1, nY+1):
            curpos = cur*cl + j  # index of the state (i,j)
            if logTriangularCoefficients[abs(i-j)] > log0:
                frompos1 = old*cl + j  # index of the state (i-1,j)
                frompos2 = cur*cl + j-1  # index of the state (i,j-1)
                frompos3 = old*cl + j-1  # index of the state (i-1,j-1)

                # We first compute the kernel value
                dist = cdists[i-1, j-1]
                gram = logTriangularCoefficients[abs(i-j)] + dist*Sig
                gram -= math.log(2 - math.exp(gram))

                # Doing the updates now, in two steps.
                aux = logP(logM[frompos1], logM[frompos2])
                logM[curpos] = logP(aux, logM[frompos3]) + gram
            else:
                logM[curpos] = log0
        # Update the column order
        cur = 1 - cur
        old = 1 - old

    aux = logM[curpos]
    return aux

def gak_d_all(X,Y,adaptive_scaling=False,**kwargs):
    return distance_matrix(X,Y,logGAK,adaptive_scaling=adaptive_scaling,**kwargs)

def gak_i_all(X,Y,adaptive_scaling=False,**kwargs):
    return distance_matrix_independent(X,Y,logGAK,adaptive_scaling=adaptive_scaling,**kwargs)

# def tslearnGAK(x, y, sigma=1.0):
#     return unnormalized_gak(x.T, y.T, sigma)

# def tslearn_gak_all(X,Y,**kwargs):
#     return distance_matrix(X,Y,tslearnGAK,kwargs)


# OLD CODE FROM RYAN -- CONTAINED BUGS
# def LGAK(x, y, sigma=0.1):
#     r"""
#     This function uses the log Global Alignment Kernel (TGAK) described in Cuturi (2011) [1]_.
#     The formula for LGAK is follows:

#     .. math::

#         LGAK(x, y,\sigma)= (\prod_{i=1}^{|\pi|}e^(\frac{1}{2\sigma^2}({x_{\pi_1(i)} - y_{\pi_2(j)}})^2+log(e^{-\frac{({x_{\pi_1(i)} - y_{\pi_2(j)}})^2}{2\sigma^2}})))
    
#     :param x: time series :code:`x`
#     :type x: np.array
#     :param y: time series :code:`x`
#     :type y: np.array
#     :param sigma: parameter of the Gaussian kernel
#     :type sigma: float
#     :return: the LGAK distance

#     """

#     cdists = cdist(x.T, y.T, "sqeuclidean")

#     Sig = 1 / (2 * sigma * sigma)
#     K = -cdists * Sig
#     K -= np.log(2 - np.exp(K))
#     K = np.exp(K)

#     sz1, sz2 = K.shape

#     cum_sum = np.zeros((sz1 + 1, sz2 + 1))
#     cum_sum[0, 0] = 1.0

#     for i in range(sz1):
#         for j in range(sz2):
#             cum_sum[i + 1, j + 1] = (
#                 cum_sum[i, j + 1] + cum_sum[i + 1, j] + cum_sum[i, j]
#             ) * K[i, j]

#     return cum_sum[sz1, sz2]