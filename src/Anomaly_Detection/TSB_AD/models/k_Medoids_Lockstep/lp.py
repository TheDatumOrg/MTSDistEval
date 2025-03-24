from turtle import distance
import numpy as np

from .dist import distance_matrix

''' X: an mts formatted (N,C,T)
    Y: an mts formatted (M,C,T)
    Return is a (N, M) distance matrix'''

def euclidean(A,B):
    return np.sqrt(np.sum((A-B)**2))

def euclidean_all(X,Y,adaptive_scaling=False,n_jobs=-1):
    n,c,t = X.shape
    m,c,t = Y.shape
    comp = n*m*c*t
    if (comp < 1e9 and (not adaptive_scaling)):
        dist = np.sqrt(np.sum((X[:,np.newaxis,...] - Y[np.newaxis,...])**2,axis=(-2,-1)))
    else:
        dist = distance_matrix(X,Y,euclidean,adaptive_scaling=adaptive_scaling)
    return dist

def lp(A,B,p):
    return np.power(np.sum(np.power(A - B, p)), 1/p)
def lp_all(X,Y, p=2,n_jobs=-1, adaptive_scaling=False):
    n,c,t = X.shape
    m,c,t = Y.shape
    comp = n*m*c*t
    if (comp < 1e9 and (not adaptive_scaling)): 
        # vectorization
        dist = np.power(np.sum(np.power(X[:,np.newaxis,...] - Y[np.newaxis,...], p), axis=(-2,-1)), 1/p)

    else:
        dist = distance_matrix(X,Y,lp,adaptive_scaling=adaptive_scaling, p=p)

    return dist

def l1(A,B):
    return np.sum(np.abs(A - B))

def l1_all(X,Y,adaptive_scaling=False,n_jobs=-1):
    n,c,t = X.shape
    m,c,t = Y.shape
    comp = n*m*c*t
    if (comp < 1e9 and (not adaptive_scaling)):
        dist = np.sum(np.abs(X[:,np.newaxis,...] - Y[np.newaxis,...]),axis=(-2,-1))
    else:
        dist = distance_matrix(X,Y,l1,adaptive_scaling=adaptive_scaling)

    return dist

def lorentzian(A,B):
    return np.sum(np.log(1+np.abs(A - B)))

def lorentzian_all(X,Y,adaptive_scaling=False,n_jobs=-1):
    n,c,t = X.shape
    m,c,t = Y.shape
    comp = n*m*c*t
    if (comp < 1e9 and (not adaptive_scaling)):
        # vectorization
        dist = np.sum(np.log(1+np.abs(X[:,np.newaxis,...] - Y[np.newaxis,...])),axis=(-2,-1))

    else:
        # To avoid OOM for large matrix
        dist = distance_matrix(X,Y,lorentzian,adaptive_scaling=adaptive_scaling)
    return dist

def avg_l1_linf(A,B):
    dist = np.sum((np.sum(np.abs(A-B),axis=1) + np.max(np.abs(A-B),axis=1)) / 2)
    return dist

def avg_l1_linf_all(X,Y,adaptive_scaling=False,n_jobs=-1):

    dist = distance_matrix(X,Y,avg_l1_linf,adaptive_scaling=adaptive_scaling)

    return dist

def jaccard(A,B):
    dist = np.sum((A-B)**2) / (np.sum(A**2 + B**2 - (A*B)) + np.finfo(float).eps)
    return dist

def jaccard_all(X,Y,adaptive_scaling=False,n_jobs=-1):

    dist_mat = distance_matrix(X,Y,jaccard,adaptive_scaling=adaptive_scaling)

    return dist_mat

def emanon4(A,B):
    dist = np.sum((A-B)**2 / (np.maximum(A,B)+np.finfo(float).eps))

    return dist

def emanon4_all(X,Y,adaptive_scaling=False,n_jobs=-1):

    dist_mat = distance_matrix(X,Y,emanon4,adaptive_scaling=adaptive_scaling) 

    return dist_mat

def soergel(A,B):
    dist = np.sum(np.abs(A-B)) / (np.sum(np.maximum(A,B))+np.finfo(float).eps)

    return dist

def soergel_all(X,Y,adaptive_scaling=False,n_jobs=-1):

    dist_mat = distance_matrix(X,Y,soergel,adaptive_scaling=adaptive_scaling) 

    return dist_mat

def topsoe(A,B):
    logAB = np.nan_to_num(np.log(A+B + np.finfo(float).eps))
    left = np.nan_to_num(np.log(2*A + np.finfo(float).eps)-logAB)
    right = np.nan_to_num(np.log(2*B + np.finfo(float).eps)-logAB)
    dist = np.sum(A * left + B * right)

    dist = np.real(dist)

    return dist

def topsoe_all(X,Y,adaptive_scaling=False,n_jobs=-1):

    dist_mat = distance_matrix(X,Y,topsoe,adaptive_scaling=adaptive_scaling) 

    return dist_mat

def clark(A,B):
    dist = np.sqrt(np.sum(np.abs(A-B)**2 / (A + B + np.finfo(float).eps)**2))
    return dist

def clark_all(X,Y,adaptive_scaling=False,n_jobs=-1):
    dist_mat = distance_matrix(X,Y,clark,adaptive_scaling=adaptive_scaling) 

    return dist_mat

def chord(A,B):
    dist = np.sum((np.nan_to_num(np.sqrt(A)) - np.nan_to_num(np.sqrt(B)))**2)
    return dist

def chord_all(X,Y,adaptive_scaling=False,n_jobs=-1):
    dist_mat = distance_matrix(X,Y,chord,adaptive_scaling=adaptive_scaling) 

    return dist_mat

def canberra(A,B):
    dist = np.sum(np.abs(A-B) / (A+B+np.finfo(float).eps))
    return dist
def canberra_all(X,Y,adaptive_scaling=False,n_jobs=-1):
    dist_mat = distance_matrix(X,Y,canberra,adaptive_scaling=adaptive_scaling) 

    return dist_mat
'''
# TEST CASE
if __name__ == "__main__":

    X = np.array([0,1,2,1]).reshape(2,1,2)
    Y = np.array([2,1,4,5]).reshape(2,1,2)

    l2_res = lp_all(X,Y,2)
    l1_res = l1_all(X,Y)
    lo_res = lorentzian_all(X,Y)

    print(l2_res)
    print(l1_res)
    print(lo_res)
'''


