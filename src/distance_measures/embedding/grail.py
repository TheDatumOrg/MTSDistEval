from GRAIL.Representation import GRAIL
from GRAIL.kshape import matlab_kshape
import numpy as np
import time
import os

from src.distance_measures.lock_step.lp import euclidean_all,lorentzian_all,l1_all
from src.distance_measures.kernel.sink import sink_d_all, sink_i_all, sink_d_denom_all, sink_i_denom_all

distances = {
    'euclidean':euclidean_all,
    'lorentzian':lorentzian_all,
    'l1':l1_all
}

def grail_embedding(X_test,X_train,sample_method='random',gamma=1,rep_length=100, adaptive_scaling=False):
    
    ALL = np.vstack((X_train,X_test))
    nc = X_train.shape[1]

    start = time.time()
    if nc > 1 and sample_method == 'random':
        # since we dont have multivariate k-shape, just randomly sample the dictionary
        # randomly select dictionary
        print("Randomly sampling dictionary")
        sampled_indices = np.random.permutation(np.arange(X_train.shape[0]))[:rep_length]
        DCT = ALL[sampled_indices]
    else:
        print("Performing kshape clustering to get dictionary")
        _, DCT = matlab_kshape(ALL.squeeze(),rep_length)
        DCT = DCT[:,np.newaxis,:]
        
    W = sink_d_all(DCT,DCT, gamma=gamma, adaptive_scaling=adaptive_scaling)
    eigval_w,eigvec_w = np.linalg.eigh(W)
    inVa = np.diag(np.power(eigval_w,-0.5))

    print("Time taken on dictionary: ", time.time()-start)

    E = sink_d_all(ALL,DCT, gamma=gamma, adaptive_scaling=adaptive_scaling)

    Zexact = E @ eigvec_w @ inVa
    Zexact = np.real(Zexact)
    Zexact = np.nan_to_num(Zexact)

    # B = np.zeros(int(rep_length/2),rep_length)
    # eigval_Z = np.linalg.eigh(Zexact.T@Zexact)
    # eigval_diag_Z = np.diag(eigval_Z)

    rep_train = np.expand_dims(Zexact[0:X_train.shape[0]],axis=1)
    rep_test = np.expand_dims(Zexact[X_train.shape[0]:],axis=1)

    return rep_test, rep_train

def grail_denom_embedding(X_test,X_train,sample_method='random',gamma=1,rep_length=100, adaptive_scaling=False):

    # ----- OFFLINE PART -----
    nc = X_test.shape[1]

    start = time.time()
    if nc > 1 or sample_method == 'random':
        # since we dont have multivariate k-shape, just randomly sample the dictionary
        # randomly select dictionary
        print("Randomly sampling dictionary")
        sampled_indices = np.random.permutation(np.arange(X_train.shape[0]))[:rep_length]
        DCT = X_train[sampled_indices]
    else:
        print("Performing kshape clustering to get dictionary")
        _, DCT = matlab_kshape(X_train.squeeze(),rep_length)
        DCT = DCT[:,np.newaxis,:]
        
    W = sink_d_denom_all(DCT,DCT, gamma=gamma, adaptive_scaling=adaptive_scaling)
    eigval_w,eigvec_w = np.linalg.eigh(W)
    inVa = np.diag(np.power(eigval_w,-0.5))

    E_train = sink_d_denom_all(X_train,DCT, gamma=gamma, adaptive_scaling=adaptive_scaling)

    offline_time = time.time()-start

    # ----- ONLINE PART -----
    E_test = sink_d_denom_all(X_test,DCT, gamma=gamma, adaptive_scaling=adaptive_scaling)
    E = np.vstack((E_train,E_test))

    Zexact = E @ eigvec_w @ inVa
    Zexact = np.real(Zexact)
    Zexact = np.nan_to_num(Zexact)

    rep_train = np.expand_dims(Zexact[0:X_train.shape[0]],axis=1)
    rep_test = np.expand_dims(Zexact[X_train.shape[0]:],axis=1)

    return rep_test, rep_train, offline_time

def get_distmat(rep_X, rep_Y, metric, **kwargs):
    metric_func = distances[metric]
    dist_mat = metric_func(rep_X,rep_Y, **kwargs)
    return dist_mat

def grail_d_all(X,Y,metric='euclidean',gamma=1,rep_length=100, adaptive_scaling=False, **kwargs):
    rep_length = min(rep_length, X.shape[0], Y.shape[0])

    rep_X, rep_Y = grail_embedding(X,Y,gamma=gamma,rep_length=rep_length, adaptive_scaling=adaptive_scaling)
    return get_distmat(rep_X, rep_Y, metric, **kwargs)

def grail_i_all(X,Y,metric='euclidean',gamma=1,rep_length=100, adaptive_scaling=False, **kwargs):
    rep_length = min(rep_length, X.shape[0], Y.shape[0])

    n,c,t = X.shape
    m,c,t = Y.shape
    X_rep = np.zeros((n,c,rep_length))
    Y_rep = np.zeros((m,c,rep_length))

    # sample_method =  'kshape' if c < 50 else 'random'
    sample_method = 'random'
    
    for i in range(c):
        print(f"Embedding channel {i}")
        Xc = X[:,i,:][:,np.newaxis,:]
        Yc = Y[:,i,:][:,np.newaxis,:]
        X_rep_c, Y_rep_c = grail_embedding(Xc,Yc,sample_method=sample_method, gamma=gamma,rep_length=rep_length, adaptive_scaling=adaptive_scaling)
        X_rep[:,i,:] = X_rep_c.squeeze()
        Y_rep[:,i,:] = Y_rep_c.squeeze()

    return get_distmat(X_rep, Y_rep, metric, **kwargs)
    
def grail_d_denom_all(X,Y,metric='euclidean',gamma=1,rep_length=100, adaptive_scaling=False, **kwargs):
    rep_length = min(rep_length, X.shape[0], Y.shape[0])

    rep_X, rep_Y, offline_time = grail_denom_embedding(X,Y,gamma=gamma,rep_length=rep_length, adaptive_scaling=adaptive_scaling)

    print("Offline time: ", offline_time)
    return get_distmat(rep_X, rep_Y, metric, **kwargs)

def grail_i_denom_all(X,Y,metric='euclidean',gamma=1,rep_length=100, adaptive_scaling=False, **kwargs):
    rep_length = min(rep_length, X.shape[0], Y.shape[0])

    n,c,t = X.shape
    m,c,t = Y.shape
    X_rep = np.zeros((n,c,rep_length))
    Y_rep = np.zeros((m,c,rep_length))

    # sample_method =  'kshape' if c < 50 else 'random'
    sample_method = 'random'

    offline_sum = 0

    for i in range(c):
        print(f"Embedding channel {i}")
        Xc = X[:,i,:][:,np.newaxis,:]
        Yc = Y[:,i,:][:,np.newaxis,:]
        X_rep_c, Y_rep_c, offline_time = grail_denom_embedding(Xc,Yc,sample_method=sample_method, gamma=gamma,rep_length=rep_length, adaptive_scaling=adaptive_scaling)
        X_rep[:,i,:] = X_rep_c.squeeze()
        Y_rep[:,i,:] = Y_rep_c.squeeze()
        offline_sum += offline_time

    print("Offline time: ", offline_sum)

    return get_distmat(X_rep, Y_rep, metric, **kwargs)