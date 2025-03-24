import numpy as np
from sktime.transformations.panel.catch22 import Catch22

from src.distance_measures.lock_step.lp import euclidean_all
from src.distance_measures.dist import distance_matrix

def catch22_all(X,Y, adaptive_scaling=False,n_jobs=1):
    c22 = Catch22()

    X_transform = np.nan_to_num(c22.fit_transform(X).to_numpy())[:, np.newaxis, :]
    Y_transform = np.nan_to_num(c22.fit_transform(Y).to_numpy())[:, np.newaxis, :]

    return euclidean_all(X_transform, Y_transform, adaptive_scaling=adaptive_scaling)

    
    