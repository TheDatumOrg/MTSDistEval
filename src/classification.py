import os
import json
import time
import sys
import csv

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src.utils import create_directory,compute_classification_metrics
from src.normalization import create_normalizer,normalization_methods
from src.onennclassifier import OneNNClassifier, MEASURES

from src.parameters import Parameters
from src.utils import multivariate as DATASETS

module = "MultivariateDistanceClassifier"

def get_optimal_parameters(metric, problem, norm, param_path):
    # Check if ensemble or not
    if "&" in metric:
        metrics = metric.split("&")
        metric_params = {}
        for m in metrics:
            local_params = get_optimal_parameters(m, problem, norm, param_path)
            metric_params[m] = local_params if local_params is not None else {}
        return metric_params

    df = pd.read_csv(param_path)

    # Get the optimal parameters for the given metric (i.e., row with highest accuracy)
    stats = df[(df.metric == metric) & (df.problem == problem) & (df.norm == norm)].sort_values('acc', ascending=False)

    if len(stats) > 0:
        dct = stats.iloc[0].metric_params

        # Convert string to dictionary
        metric_params = json.loads(dct.replace("'", "\""))

        print(f"Using optimal parameters for metric {metric} and problem {problem}: {metric_params}")

        return metric_params
    else:
        print(f"No optimal parameters found for metric {metric} and problem {problem}, expecting that they are not needed.")
        return None

def main(params:Parameters):
    data_path = params.data_path
    OUTDIR = params.output_path
    metric = params.metric
    problem_idx = params.problem_idx
    problem = params.problem
    norm = params.norm
    itr = params.itr
    save_distances = params.save_distances
    metric_params = params.metric_params
    param_path = params.param_path
    run_type = params.run_type
    n_jobs = params.n_jobs
    testrun = params.testrun
    parameters_given = params.metric_params not in [None, {}, '']

    if run_type == 'inference':
        OUTDIR += "/inference"
        outpath = os.path.join(OUTDIR, 'classification_results.csv')
    else:
        OUTDIR += "/validation"
        outpath = os.path.join(OUTDIR, 'validation_results.csv')

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    print("=======================================================================")
    print("[{}] Starting Classification Experiment".format(module))
    print("=======================================================================")
    print("[{}] Data path: {}".format(module, data_path))
    print("[{}] Output Dir: {}".format(module, outpath))
    print("[{}] Iteration: {}".format(module, itr))
    print("[{}] Problem: {} | {}".format(module, problem_idx, problem))
    print("[{}] Metric: {}".format(module, metric))
    print("[{}] Normalisation: {}".format(module, norm))
    print("[{}] Run Type: {}".format(module, run_type))

    #Call Datasets
    print("[{}] Loading data".format(module))
    
    if testrun: # Generate dummy data
        X_train = np.random.randn(10,3,32)
        y_train = np.random.randint(0, 2, size=(10,))
        X_test = np.random.randn(10,3,32)
        y_test = np.random.randint(0, 2, size=(10,))
    else:
        X_train = np.load(os.path.join(data_path, problem, f'{problem}_train_X.npy'))
        y_train = np.load(os.path.join(data_path, problem, f'{problem}_train_Y.npy'))

        if run_type == "inference":
            X_test  = np.load(os.path.join(data_path, problem, f'{problem}_test_X.npy'))
            y_test  = np.load(os.path.join(data_path, problem, f'{problem}_test_Y.npy'))
        else:
            X_test = X_train
            y_test = y_train

    # If UTS, then fix
    if len(X_train.shape) == 2:
        X_train = X_train[:,np.newaxis,:]
        X_test = X_test[:,np.newaxis,:]

    # If parameters not given, check for optimal parameters
    if run_type == 'inference' and not parameters_given and param_path is not None:
        metric_params = get_optimal_parameters(metric, problem, norm, param_path)

    if metric_params is None:
        metric_params = {}

    print("[{}] Parameters: {}".format(module, metric_params))

    #Create Normalizer & Normalize Data
    print("[{}] X_train: {}".format(module, X_train.shape))
    print("[{}] X_test: {}".format(module, X_test.shape))

    if norm in ['none', 'adaptive']:
        X_train_norm = X_train
        X_test_norm = X_test
    else:
        normalizer = create_normalizer(norm)
        X_train_norm = normalizer.transform(X_train)
        X_test_norm = normalizer.transform(X_test)

    #Normalize Labels
    print("[{}] y_train: {}".format(module, y_train.shape))
    print("[{}] y_test: {}".format(module, y_test.shape))
    label_encode = LabelEncoder()
    y_train_norm = label_encode.fit_transform(y_train)
    y_test_norm = label_encode.transform(y_test)

    onenn = OneNNClassifier(metric = metric, metric_params=metric_params, n_jobs=n_jobs, adaptive_scaling=norm == 'adaptive')

    # Check if distance matrix is saved
    DISTMATDIR = os.path.join(OUTDIR, 'distance_matrices')
    os.makedirs(DISTMATDIR, exist_ok=True)
    distmat_path = os.path.join(DISTMATDIR, f'{metric}_{problem}_{norm}_{metric_params}_{run_type}.npy')
    distmat = None
    if os.path.exists(distmat_path):
        distmat = np.load(distmat_path)
        print(f"Loaded distance matrix from {distmat_path}")

    start_t = time.time()

    onenn.fit(X_train_norm,y_train_norm)
    pred, distmat = onenn.predict(X_test_norm, self_similarity=run_type == 'inference', distance_matrix=distmat, testrun=testrun)

    end_t = time.time()
    runtime = end_t - start_t
    
    # Save distance matrix
    if save_distances:
        print(f"Saving distance matrix to {distmat_path}")
        np.save(distmat_path, distmat)

    params = {
        'norm':norm,
        'metric':metric,
        'input_path':data_path,
        'parameters_given':parameters_given
    }
    
    if metric_params is not None:
        params['metric_params'] = str(metric_params)
    else:
        params['metric_params'] = None
    
    params['runtime'] = runtime
    results = compute_classification_metrics(y_test_norm,pred)

    # Add metric, norm, problem, problem_idx, iteration to results
    results['problem'] = problem
    results['problem_idx'] = problem_idx
    results['itr'] = itr
    
    # Add parameters to results
    results.update(params)

    print(f"Runtime: {runtime:.4f} sec")
    print("Results:")
    for k,v in results.items():
        print(f"{k}: {v}")

    columns = ["input_path", "acc","precision","recall","f1","problem","problem_idx","itr","norm","metric","metric_params","runtime", "parameters_given"]

    # Save results
    exists = os.path.exists(outpath)
    with open(outpath, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(results)

if __name__ == "__main__":
    print("Arguments:", sys.argv)
    if len(sys.argv) == 1:
        sys.argv = [
            "main.py",
            "-mp", "inference",
            "-d", "UEA_archive",
            "-m", "l2",
            "-p", "BasicMotions",
            "-n", "none",
        ]

    params = Parameters.parse(sys.argv[1:])

    # Check if it is a singleton run or a combined run
    if "*" in params.problem:
        start = params.problem.replace("*", "")

        subset = DATASETS[DATASETS.index(start):] if start in DATASETS else DATASETS

        for problem in subset:
            params.problem = problem
            try:
                main(params)
            except Exception as e:
                print(f"Error in problem {problem}: {e}")
                continue
                
    else:
        main(params)