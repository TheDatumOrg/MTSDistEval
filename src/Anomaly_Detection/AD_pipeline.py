import pandas as pd
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
import os
import argparse
import torch
import random
import time
import numpy as np
# seeding
seed = 2024
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=False, default="TSB-AD-M")
    parser.add_argument('-f','--folder', required=True)
    parser.add_argument('-s','--save_path', required=True)
    parser.add_argument('-sw','--slidingWindow', required=False, default=100)
    parser.add_argument('-ws','--slidingWindowStrategy', required=False, default="auto")
    parser.add_argument('-st','--stride', required=False, default=1)
    parser.add_argument('-dm','--distance_measure', required=False, default='euclidean')
    parser.add_argument('-nn','--n_neighbors', required=False, default=1)
    parser.add_argument('-m','--method', required=False, default='largest')
    parser.add_argument('-n','--normalize', required=False, default=False)
    parser.add_argument('-j','--n_jobs', required=False, default=-1)
    arguments = parser.parse_args()
    PATH = arguments.path
    folder = arguments.folder
    save_path = arguments.save_path
    slidingWindow = int(arguments.slidingWindow)
    slidingWindowStrategy = arguments.slidingWindowStrategy
    stride = int(arguments.stride)
    distance_measure = arguments.distance_measure
    n_neighbors = int(arguments.n_neighbors)
    method = arguments.method
    normalize = arguments.normalize
    n_jobs = int(arguments.n_jobs)
    print(f'folder: {folder}, distance: {distance_measure}, normalize: {normalize}', flush=True)
    # Specify Anomaly Detector to use and data directory
    AD_Name = 'KNN_multivariate_sliding'
    data_direc = os.path.join(PATH, folder)
    # Loading Data
    df = pd.read_csv(data_direc).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    s_time = time.time()

    if slidingWindowStrategy == "auto":

        window_lens = []
        for c in range(data.shape[1]):
            window_lens.append(find_length_rank(data[:,c], rank=1))
        slidingWindow = np.maximum(int(np.mean(window_lens)), stride)

    print(f"Raw data shape {data.shape} | Sliding Window: {slidingWindow} | Stride: {stride}", flush=True)

    # Applying Anomaly Detector
    output = run_Unsupervise_AD(AD_Name, data, slidingWindow = slidingWindow, stride=stride, distance_measure = distance_measure, \
                                n_neighbors=n_neighbors, method=method, normalize=normalize, n_jobs=n_jobs)
    # print('output:', output)
    pred_e_time = time.time()
    # Evaluation
    evaluation_result = get_metrics(output, label)
    evaluation_result = pd.DataFrame([evaluation_result])
    evaluation_result['PATH'] = PATH
    evaluation_result['problem'] = folder
    evaluation_result['slidingWindow'] = slidingWindow
    evaluation_result['stride'] = stride
    evaluation_result['distance_measure'] = distance_measure
    evaluation_result['n_neighbors'] = n_neighbors
    evaluation_result['method'] = method
    evaluation_result['normalize'] = normalize
    evaluation_result['AD_Name'] = AD_Name
    evaluation_result['time'] = round(pred_e_time - s_time, 4)
    print(f'{evaluation_result}', flush=True)

    if slidingWindowStrategy == "auto":
        slidingWindow = "auto"
    else:
        slidingWindow = slidingWindow
    dir_path = os.path.join(save_path, f"{distance_measure}/Exp_win-{slidingWindow}_strd-{stride}_norm-{normalize}", folder[:-4])

    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    # Save the array to the file
    csv_filename = "evaluation_AD.csv"
    evaluation_result.to_csv(os.path.join(dir_path, csv_filename), index=False)
    e_time = time.time()
    print(f"Total time: {e_time -  s_time:.2f}s", flush=True)
# OK