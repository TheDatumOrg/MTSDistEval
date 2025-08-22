import numpy as np
import os
import multiprocessing
from numba import set_num_threads


from src.distance_measures.lock_step.lp import lp_all,l1_all,lorentzian_all,euclidean_all,avg_l1_linf_all,jaccard_all,emanon4_all,soergel_all,topsoe_all,clark_all,chord_all,canberra_all
from src.distance_measures.elastic.dtw import dtw_all
from src.distance_measures.sliding.sbd_numba_rocket_test import SBD_Local_all_rocket, SBD_Global_all_rocket
from src.distance_measures.elastic.lcss import lcss_all
from src.distance_measures.elastic.erp import erp_all
from src.distance_measures.elastic.twe import twe_all
from src.distance_measures.elastic.msm import msm_all
from src.distance_measures.kernel.rbf import rbf_all
from src.distance_measures.kernel.gak import gak_d_all, gak_i_all
from src.distance_measures.kernel.sink import *
from src.distance_measures.kernel.kdtw import kdtw_d_all, kdtw_i_all
from src.distance_measures.embedding.pca import pca_all, eros_all
from src.distance_measures.model.hmm import hmmdiv_d_all, hmmdiv_i_all
from src.distance_measures.model.kl import *
from src.distance_measures.feature.catch22 import catch22_all
from src.distance_measures.feature.tsfresh import tsfresh_all
from src.distance_measures.embedding.grail import *
from src.distance_measures.embedding.ts2vec import ts2vec_d_all, ts2vec_i_all
from src.distance_measures.embedding.tloss import tloss_d_all, tloss_i_all

MEASURES = {
    'l2': lambda X,Y,**kwargs: lp_all(X,Y,2, **kwargs),
    'l1':lambda X,Y,**kwargs:l1_all(X,Y,**kwargs),
    'lorentzian':lambda X,Y,**kwargs:lorentzian_all(X,Y,**kwargs),
    'euclidean':lambda X,Y,**kwargs:euclidean_all(X,Y,**kwargs),
    'avg_l1_inf':lambda X,Y,**kwargs:avg_l1_linf_all(X,Y,**kwargs),
    'jaccard':lambda X,Y,**kwargs:jaccard_all(X,Y,**kwargs),
    'emanon4':lambda X,Y,**kwargs:emanon4_all(X,Y,**kwargs),
    'soergel':lambda X,Y,**kwargs:soergel_all(X,Y,**kwargs),
    'topsoe':lambda X,Y,**kwargs:topsoe_all(X,Y,**kwargs),
    'clark':lambda X,Y,**kwargs:clark_all(X,Y,**kwargs),
    'chord':lambda X,Y,**kwargs:chord_all(X,Y,**kwargs),
    'canberra':lambda X,Y,**kwargs:canberra_all(X,Y,**kwargs),
    'dtw-d': lambda X,Y,**kwargs: dtw_all(X,Y,'dependent',**kwargs),
    'dtw-i': lambda X,Y,**kwargs: dtw_all(X,Y,'independent',**kwargs),
    'lcss-d': lambda X,Y,**kwargs: lcss_all(X,Y,'dependent',**kwargs),
    'lcss-i': lambda X,Y,**kwargs: lcss_all(X,Y,'independent',**kwargs),
    'erp-d': lambda X,Y,**kwargs: erp_all(X,Y,'dependent',**kwargs),
    'erp-i': lambda X,Y,**kwargs: erp_all(X,Y,'independent',**kwargs),
    'twe-d': lambda X,Y,**kwargs: twe_all(X,Y,'dependent',**kwargs),
    'twe-i': lambda X,Y,**kwargs: twe_all(X,Y,'independent',**kwargs),
    'msm-d': lambda X,Y,**kwargs: msm_all(X,Y,'dependent',**kwargs),
    'msm-i': lambda X,Y,**kwargs: msm_all(X,Y,'independent',**kwargs),
    'rbf': lambda X,Y,**kwargs: rbf_all(X,Y, **kwargs),
    'gak-d': lambda X,Y,**kwargs: gak_d_all(X,Y,**kwargs),
    'gak-i': lambda X,Y,**kwargs: gak_i_all(X,Y,**kwargs),
    'sink-d': lambda X,Y,**kwargs: sink_d_all(X,Y,**kwargs),
    'sink-d-denom': lambda X,Y,**kwargs: sink_d_denom_all(X,Y,**kwargs),
    'sink-i': lambda X,Y,**kwargs: sink_i_all(X,Y,**kwargs),
    'sink-i-denom': lambda X,Y,**kwargs: sink_i_denom_all(X,Y,**kwargs),
    'kdtw-d': lambda X,Y,**kwargs: kdtw_d_all(X,Y,**kwargs),
    'kdtw-i': lambda X,Y,**kwargs: kdtw_i_all(X,Y,**kwargs),
    'kl-d': kl_d_all,
    'kl-i': kl_i_all,
    'pca': lambda X,Y,**kwargs: pca_all(X,Y,**kwargs),
    'eros': lambda X,Y, **kwargs: eros_all(X,Y, **kwargs),
    'sbd-i': lambda X,Y,**kwargs: SBD_Local_all_rocket(X,Y,**kwargs),
    'sbd-d': lambda X,Y,**kwargs: SBD_Global_all_rocket(X,Y,**kwargs),
    'hmm-rescale-d': lambda X,Y,**kwargs: hmmdiv_d_all(X,Y, normalize_probs=True,**kwargs),
    'hmm-rescale-i': lambda X,Y,**kwargs: hmmdiv_i_all(X,Y, normalize_probs=True,**kwargs),
    'catch22-i': lambda X,Y,**kwargs: catch22_all(X,Y,**kwargs),
    'grail-d': lambda X,Y,**kwargs: grail_d_all(X,Y,**kwargs),
    'grail-d-denom': lambda X,Y,**kwargs: grail_d_denom_all(X,Y,**kwargs),
    'grail-i': lambda X,Y,**kwargs: grail_i_all(X,Y,**kwargs),
    'grail-i-denom': lambda X,Y,**kwargs: grail_i_denom_all(X,Y,**kwargs),
    'ts2vec-d': lambda X,Y,**kwargs: ts2vec_d_all(X,Y,**kwargs),
    'ts2vec-i': lambda X,Y,**kwargs: ts2vec_i_all(X,Y,**kwargs),
    'tloss-d': lambda X, Y, y_labels, **kwargs: tloss_d_all(X, Y, y_labels, **kwargs),
    'tloss-i': lambda X, Y, y_labels, **kwargs: tloss_i_all(X, Y, y_labels, **kwargs),
}

class OneNNClassifier:
    def __init__(self,metric,metric_params={},adaptive_scaling=False,n_jobs=-1):
        # Check if metric is valid
        metric = metric.lower()
        self.metric = metric
        self.metric_params = metric_params
        self.adaptive_scaling=adaptive_scaling
        self.n_jobs = n_jobs

        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = min(multiprocessing.cpu_count(), 9)
        else:
            n_jobs = self.n_jobs

        # super raises numba import exception if not available
        # so now we know we can use numba

        set_num_threads(n_jobs)

        def tsfresh(X,Y,**kwargs):
            kwargs['n_jobs'] = n_jobs
            return tsfresh_all(X,Y,**kwargs)

        MEASURES['tsfresh-i'] = tsfresh
        
        for m in metric.split("&"):
            if m not in MEASURES:
                raise ValueError(f"Invalid metric {metric}. Must be one of {list(MEASURES.keys())}")

    def fit(self,X,y):
        self._X = X
        self._y = y

    def distance_matrix(self, X, metric, metric_params):
        if "&" in metric: # Ensemble
            # Get all components
            components = metric.split("&")
            distance_matrix = np.zeros((X.shape[0], self._X.shape[0]))
            for component in components:
                print(f"Calculating distance matrix for component {component}")

                assert component in metric_params.keys(), f"Component {component} not found in metric_params"

                local_dm = self.distance_matrix(X, component, metric_params[component]) # recursive call

                # Perform min-max scaling
                local_dm = (local_dm - np.nanmin(local_dm)) / (np.nanmax(local_dm) - np.nanmin(local_dm))
                distance_matrix += local_dm
        else: # Single metric
            if metric.startswith('tloss'):
                distance_matrix = MEASURES[metric](X, self._X, self._y, adaptive_scaling=self.adaptive_scaling, n_jobs=self.n_jobs, **metric_params)
            else:
                distance_matrix = MEASURES[metric](X,self._X, adaptive_scaling=self.adaptive_scaling, n_jobs=self.n_jobs, **metric_params)

        return distance_matrix


    def predict(self, X, self_similarity=True, distance_matrix=None, **kwargs):
        # X is test set; self._X is train set
        if not distance_matrix:
            if kwargs.get('testrun', False):
                distance_matrix = np.abs(np.random.randn(X.shape[0], self._X.shape[0]))
            else:
                distance_matrix = self.distance_matrix(X, self.metric, self.metric_params)


        print("Number of Nan/Inf: ", sum(np.isnan(distance_matrix.reshape(-1))), sum(np.isinf(distance_matrix.reshape(-1))))
        # Fill the diagonal with nans if no self similarity
        if not self_similarity:
            np.fill_diagonal(distance_matrix,np.nan)

        if self.metric.startswith(('kdtw', 'gak', 'sink')):
            ind = np.nanargmax(distance_matrix,axis=1)
        else:
            ind = np.nanargmin(distance_matrix,axis=1)
        ind = ind.T
        pred = self._y[ind]
        return pred, distance_matrix