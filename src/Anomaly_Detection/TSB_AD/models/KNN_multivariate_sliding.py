"""
This function is adapted from [pyod] by [yzhao062]
Original source: [https://github.com/yzhao062/pyod]
"""

from __future__ import division
from __future__ import print_function
from warnings import warn

import numpy as np
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
import math
from .k_Medoids_Elastic.dtw import dtw_all
from .k_Medoids_Elastic.lcss import lcss_all
from .k_Medoids_Elastic.erp import erp_all
from .k_Medoids_Elastic.twe import twe_all
from .k_Medoids_Elastic.msm import msm_all
from .k_Medoids_SBD.sbd_numba_rocket_test import SBD_Local_all_rocket, SBD_Global_all_rocket
from .k_Medoids_Lockstep.lp import euclidean_all, lorentzian_all
from .base import BaseDetector
from ..utils.utility import zscore
class Window:
    """ 
    The class for rolling window feature mapping for multivariate time series (MTS).
    The mapping converts the original MTS X into a 3D array. 
    The result consists of rows of sliding windows of the original X. 
    """
    def __init__(self, window=100, stride=1):
        self.window = window
        self.stride = stride
        self.detector = None
    
    def convert(self, X):
        """
        Converts the multivariate time series into sliding window segments.
        
        Parameters:
            X (numpy.ndarray): Input MTS with shape (L, C).
        
        Returns:
            numpy.ndarray: Output matrix of shape (num_windows, window, C).
        """
        if len(X.shape) != 2:
            raise ValueError("Input X must be a 2D array with shape (L, C).")
        
        L, C = X.shape
        n = self.window
        # print(f"n: {n}")
        if n == 0:
            return X  # Return the original time series if the window size is 0
        
        subsequences = []
        # Create sliding windows
        for start_idx in range(0, L - n + 1, self.stride):
            end_idx = start_idx + n
            subsequences.append(X[start_idx:end_idx, :])
        print(f"Sliding window shape: {np.array(subsequences).shape}")
        return np.array(subsequences)

class KNN(BaseDetector):
    # noinspection PyPep8
    """kNN class for outlier detection.
    For an observation, its distance to its kth nearest neighbor could be
    viewed as the outlying score. It could be viewed as a way to measure
    the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
    details.

    Three kNN detectors are supported:
    largest: use the distance to the kth neighbor as the outlier score
    mean: use the average of all k neighbors as the outlier score
    median: use the median of the distance to k neighbors as the outlier score

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_neighbors : int, optional (default = 10)
        Number of neighbors to use by default for k neighbors queries.

    method : str, optional (default='largest')
        {'largest', 'mean', 'median'}

        - 'largest': use the distance to the kth neighbor as the outlier score
        - 'mean': use the average of all k neighbors as the outlier score
        - 'median': use the median of the distance to k neighbors as the
          outlier score

    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for `radius_neighbors`
        queries.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

        .. deprecated:: 0.74
           ``algorithm`` is deprecated in PyOD 0.7.4 and will not be
           possible in 0.7.6. It has to use BallTree for consistency.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'precomputed'

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only kneighbors and kneighbors_graph methods.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """
    def __init__(self, slidingWindow=100, stride = 1, sub=True, contamination=0.1, n_neighbors=10, method='largest',
                 radius=1.0, algorithm='auto', leaf_size=30,
                 metric='precomputed', p=2, metric_params=None, n_jobs=1, normalize=False,
                 **kwargs):
                
        self.slidingWindow = slidingWindow
        self.stride = stride
        self.sub = sub
        self.n_neighbors = n_neighbors
        self.method = method
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.normalize = normalize
        self.n_jobs = n_jobs

        if self.algorithm != 'auto' and self.algorithm != 'ball_tree':
            warn('algorithm parameter is deprecated and will be removed '
                 'in version 0.7.6. By default, ball_tree will be used.',
                 FutureWarning)
            
        self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                       radius=self.radius,
                                       algorithm=self.algorithm,
                                       leaf_size=self.leaf_size,
                                       metric=self.metric,
                                       p=self.p,
                                       metric_params=self.metric_params,
                                       n_jobs=self.n_jobs,
                                       **kwargs)

    def fit(self, X, distance_measure = 'euclidean',y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        if self.normalize: X = zscore(X, axis=0, ddof=1)
        else: print("Not Normalized")

        if self.sub:
            # Converting time series data into matrix format
            X = Window(window = self.slidingWindow, stride = self.stride).convert(X)
        # if self.normalize: X = zscore(X, axis=1, ddof=1)

        # validate inputs X and y (optional)
        # X = check_array(X)
        X = np.transpose(X, axes=(0, 2, 1))
        if distance_measure == 'euclidean':
            distance_matrix = euclidean_all(X, X) # X shape: (bs, channel, length)
        elif distance_measure == "Lorentzian":
            distance_matrix = lorentzian_all(X, X)
        elif distance_measure == "DTW_D":
            distance_matrix = dtw_all(X, X, mode='dependent', sakoe_chiba_radius=None, itakura_max_slope=None)
        elif distance_measure == "DTW_I":
            distance_matrix = dtw_all(X, X, mode='independent', sakoe_chiba_radius=None, itakura_max_slope=None)
        elif distance_measure == "LCSS_D":
            distance_matrix = lcss_all(X, X, mode='dependent', epsilon=0.5, sakoe_chiba_radius=0.1)
        elif distance_measure == "LCSS_I":
            distance_matrix = lcss_all(X, X, mode='independent', epsilon=1.0, sakoe_chiba_radius=0.05)
        elif distance_measure == "ERP_D":
            distance_matrix = erp_all(X, X, mode='dependent')
        elif distance_measure == "ERP_I":
            distance_matrix = erp_all(X, X, mode='independent')
        elif distance_measure == "TWE_D":
            distance_matrix = twe_all(X, X, mode='dependent', lmbda=1.0, nu=0.0001)
        elif distance_measure == "TWE_I":
            distance_matrix = twe_all(X, X, mode='independent', lmbda=0.5, nu=0.01)
        elif distance_measure == "MSM_D":
            distance_matrix = msm_all(X, X, mode='dependent', c=0.5)
        elif distance_measure == "MSM_I":
            distance_matrix = msm_all(X, X, mode='independent', c=0.5)
        elif distance_measure == "SBD_D":
            distance_matrix = SBD_Global_all_rocket(X, X)
        elif distance_measure == "SBD_I":
            distance_matrix = SBD_Local_all_rocket(X, X)

        else:
            raise ValueError(f"Sorry, wrong distance measure name.")

        np.fill_diagonal(distance_matrix, 1e10)
        distance_matrix[np.isnan(distance_matrix)] = 1e10
        distance_matrix[np.isclose(distance_matrix, 0, atol=1e-10)] = 0

        self.neigh_.fit(distance_matrix)
        # Find the nearest neighbors for all points
        # k=2 means find the closest 2 neighbors
        dist_arr, _  = self.neigh_.kneighbors(distance_matrix, n_neighbors=self.n_neighbors, return_distance=True)
        # print(f'dist_arr: {dist_arr[dist_arr != 0]}')
        self.decision_scores_ = self._get_dist_by_method(dist_arr)
        # padded decision_scores_
        if self.decision_scores_.shape[0] < n_samples:
            loss_win = self.decision_scores_
            test_score = np.zeros(n_samples)
            count = np.zeros(n_samples)
            for i, score in enumerate(loss_win):
                start = i * self.stride
                end = start + self.slidingWindow
                test_score[start:end] += score
                count[start:end] += 1
            test_score = test_score / np.maximum(count, 1)

            # self.decision_scores_ = np.array([self.decision_scores_[0]]*math.ceil((self.slidingWindow-1)/2) + 
            #             list(self.decision_scores_) + [self.decision_scores_[-1]]*((self.slidingWindow-1)//2))
            self.decision_scores_ = test_score
        return self

    # This function is used only for realize abstract method decision_function, not useful for our purpose.
    def decision_function(self, X):
        pass

    def _get_dist_by_method(self, dist_arr):
        """Internal function to decide how to process passed in distance array

        Parameters
        ----------
        dist_arr : numpy array of shape (n_samples, n_neighbors)
            Distance matrix.

        Returns
        -------
        dist : numpy array of shape (n_samples,)
            The outlier scores by distance.
        """
        if self.method == 'largest':
            return dist_arr[:, -1]
        elif self.method == 'mean':
            return np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            return np.median(dist_arr, axis=1)
# OK