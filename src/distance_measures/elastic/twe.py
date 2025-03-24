import numpy as np
from src.distance_measures.dist import local_alignment
from sktime.distances import twe_distance

from src.distance_measures.elastic.lower_bounding import resolve_bounding_matrix

from src.distance_measures.elastic.twe_numba import twe, twe_d_dist_mat,twe_i_dist_mat, pad_ts

def twe_all(X,Y,mode='dependent',sakoe_chiba_radius=None,itakura_max_slope=None,nu=0.001,lmbda=1.0,p=2, adaptive_scaling=False, **kwargs):
    if mode == 'dependent':
        nu = nu
        lmbda = lmbda * np.sqrt(X.shape[1])
        
        bounding_matrix = resolve_bounding_matrix(X[0],Y[0],window=sakoe_chiba_radius,itakura_max_slope=itakura_max_slope)
        dist_mat = twe_d_dist_mat(X,Y,bounding_matrix,lmbda,nu,p,adaptive_scaling, **kwargs)
    elif mode =='independent':
        bounding_matrix = resolve_bounding_matrix(X[0,[0],:],Y[0,[0],:],window = sakoe_chiba_radius,itakura_max_slope = itakura_max_slope)
        dist_mat = twe_i_dist_mat(X,Y,bounding_matrix,lmbda,nu,p,adaptive_scaling, **kwargs)
    return dist_mat

def twe_d(A, B, nu, lmbda):
    return twe_distance(A, B, nu=nu, lmbda=lmbda, p=2)

def twe_i(A,B, nu, lmbda):
    return local_alignment(A, B, twe_distance, nu=nu, lmbda=lmbda)

"""
OLD IMPLEMENTATION

def twe_d(A, B, nu=0.001, lmbda=4/9, degree=2):
    # [distance, DP] = TWED( A, B, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given (multivariate) time series A and B
    #
    # A      := Time series A (shape = (n_dimensions, n_samples))
    # B      := Time series B (shape = (n_dimensions, n_samples))
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # degree := Degree of the p norm for local cost.
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # Transpose to match code
    A = A.T
    B = B.T

    n = A.shape[0]
    m = B.shape[0]

    # Add extra dimension if UTS
    if len(A.shape) == 1:
        A = A[:, np.newaxis]
        B = B[:, np.newaxis]

    timeSA = np.arange(n)
    timeSB = np.arange(m)

    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, 1:] = np.inf
    DP[1:, 0] = np.inf

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):

            # Deletion in A
            del_a = (
                DP[i - 1, j]
                + Dlp(A[i - 1], A[i], p=degree)
                + nu * (timeSA[i] - timeSA[i - 1])
                + lmbda
            )

            # Deletion in B
            del_b = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j], p=degree)
                + nu * (timeSB[j] - timeSB[j - 1])
                + lmbda
            )

            # Keep data points in both time series
            match = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j], p=degree)
                + Dlp(A[i - 1], B[j - 1], p=degree)
                + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
            )

            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = min(del_a, del_b, match)

    distance = DP[n-1, m-1]
    return distance

def twe_i(A,B, nu=0.001, lmbda=4/9, degree=2):
    return local_alignment(A, B, twe_d, nu=nu, lmbda=lmbda, degree=degree)

def backtracking(DP):
    # [ best_path ] = BACKTRACKING ( DP )
    # Compute the most cost efficient path
    # DP := DP matrix of the TWED function

    x = np.shape(DP)
    i = x[0] - 1
    j = x[1] - 1

    # The indices of the paths are save in opposite direction
    # path = np.ones((i + j, 2 )) * np.inf;
    best_path = []

    steps = 0
    while i != 0 or j != 0:
        best_path.append((i - 1, j - 1))

        C = np.ones((3, 1)) * np.inf

        # Keep data points in both time series
        C[0] = DP[i - 1, j - 1]
        # Deletion in A
        C[1] = DP[i - 1, j]
        # Deletion in B
        C[2] = DP[i, j - 1]

        # Find the index for the lowest cost
        idx = np.argmin(C)

        if idx == 0:
            # Keep data points in both time series
            i = i - 1
            j = j - 1
        elif idx == 1:
            # Deletion in A
            i = i - 1
            j = j
        else:
            # Deletion in B
            i = i
            j = j - 1
        steps = steps + 1

    best_path.append((i - 1, j - 1))

    best_path.reverse()
    return best_path[1:]
""" 