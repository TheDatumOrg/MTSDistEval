import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from .utils.slidingWindows import find_length_rank

Unsupervise_AD_Pool = ['KNN_multivariate_sliding']

def run_Unsupervise_AD(model_name, data, **kwargs):
    # try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    # except:
    #     error_message = f"Model function '{function_name}' is not defined."
    #     print(error_message)
    #     return error_message

def run_KNN_multivariate_sliding(data, slidingWindow=100, stride=1, n_neighbors=10, method='largest', normalize=False, n_jobs=1, distance_measure = 'euclidean'):
    from .models.KNN_multivariate_sliding import KNN
    clf = KNN(slidingWindow=slidingWindow, stride=stride, n_neighbors=n_neighbors,method=method, n_jobs=n_jobs, normalize=normalize)
    clf.fit(data, distance_measure = distance_measure)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score
# OK