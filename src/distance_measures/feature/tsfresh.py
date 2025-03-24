from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

import numpy as np

from src.distance_measures.lock_step.lp import euclidean_all
from src.distance_measures.dist import distance_matrix

def tsfresh_all(X,Y, n_jobs=1,adaptive_scaling=False):
    tsfresh = TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False, n_jobs=n_jobs)

    Y_transform = np.nan_to_num(tsfresh.fit_transform(Y, n_jobs))[:, np.newaxis, :]
    X_transform = np.nan_to_num(tsfresh.transform(X, n_jobs))[:, np.newaxis, :]

    return euclidean_all(X_transform, Y_transform, adaptive_scaling=adaptive_scaling)