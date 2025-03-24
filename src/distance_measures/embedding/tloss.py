from src.libraries.tloss.encode import encode
from src.distance_measures.lock_step.lp import euclidean_all
import numpy as np

def tloss_d_all(X_test, X_train, y_train, adaptive_scaling, n_jobs=None, **kwargs):
    train_repr, test_repr = encode(X_train=X_train, y_train=y_train, X_test=X_test, **kwargs)

    # Add a channel dimension
    train_repr = train_repr[:,np.newaxis,:]
    test_repr = test_repr[:,np.newaxis,:]

    return euclidean_all(test_repr, train_repr, adaptive_scaling=adaptive_scaling)

def tloss_i_all(X_test, X_train, y_train, adaptive_scaling, n_jobs=None, **kwargs):

    rep_length = 320
    max_channels = 100

    n,c,t = X_train.shape
    m,c,t = X_test.shape

    c = min(c, max_channels)

    X_train_rep = np.zeros((n,c,rep_length))
    X_test_rep = np.zeros((m,c,rep_length))

    # Embed each channel independently
    for i in range(c):
        print(f"Embedding channel {i}")
        Xc = X_train[:,i,:][:,np.newaxis,:]
        Yc = X_test[:,i,:][:,np.newaxis,:]
        X_rep_c, Y_rep_c = encode(X_train=Xc, y_train=y_train, X_test=Yc, **kwargs)
        X_train_rep[:,i,:] = X_rep_c
        X_test_rep[:,i,:] = Y_rep_c

    # Calculate distance matrix
    return euclidean_all(X_test_rep, X_train_rep, adaptive_scaling=adaptive_scaling)


