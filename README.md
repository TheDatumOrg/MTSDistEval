# Multivariate Time-Series Distances Evaluation

## 1. Setting up the environment:

```shell
pip install -r requirements.txt
```

## 2. Getting the data:
- The original UEA archive, preprocessed to have equal-length time-series can be downloaded from the following [link](https://drive.google.com/file/d/1XWgiN0-1K3HaC4CWHvQjTHVdVhajMzH_/view?usp=sharing).
- The downsampled version of the UEA archive (used to evaluate elastic measures) can be downloaded from [here](https://drive.google.com/file/d/1gYPohWLcpYWuTf9mTdyDl0TfH2aXyfV5/view?usp=drive_link).
- The TSB-AD-M archive can be downloaded from [here](https://www.thedatum.org/datasets/TSB-AD-M.zip), with the related repository available [here](https://github.com/TheDatumOrg/TSB-AD).

## 3. Running classification experiments:
You can run either inference or LOOCV validation on a specific dataset with a specific metric and normalization method through running the [src/classification.py](src/classification.py) script, with the following arguments:
- `-mp` - run type (options: inference, loocv)
- `-d` or `--data` - path to the data directory
- `-p` or `--problem` - name of the dataset to run classification on (e.g. BasicMotions)
- `-m` or `--measure` - name of the measure to use (e.g. l2)
- `-n` or `--norm` - name of the normalization method to use (options: zscore, minmax, median, mean, unit, sigmoid, tanh, none)
- `-c` or `--metric_params` - additional parameters for the metric (e.g. gamma for SINK kernel), passed as key=value pairs separated by spaces


Example 1: Run inference with Euclidean distance on the BasicMotions dataset with z-score normalization we would run:
```shell
python -m src.classification -mp inference -d $DATASET_DIR$ -p BasicMotions -m l2 -n zscore-i
```
Example 2: Run inference with DTW-D distance on the BasicMotions dataset with z-score normalization we would run:
```shell
python -m src.classification -mp inference -d $DATASET_DIR$ -p BasicMotions -m dtw-d -n zscore-i -c sakoe_chiba_radius=0.1
```

## 4. Example Usage
This repository primarily focuses on benchmarking experiments, and an MTS Distance library will be released soon. To compute MTS distances using this code, please refer to the example as bellow. A complete list of supported MTS distances is available in `src/onennclassifier.py`.
```shell
import numpy as np
from src.onennclassifier import MEASURES

euc_dist  = MEASURES['l2']
sbd_indep = MEASURES['sbd-i']
sbd_dep   = MEASURES['sbd-d']
dtw_indep = MEASURES['dtw-i']
dtw_dep   = MEASURES['dtw-d']

X = np.random.rand(1, 2, 50) # shape: batchsize x n_channel x n_timesteps
Y = np.random.rand(1, 2, 50) # shape: batchsize x n_channel x n_timesteps

result_euc = euc_dist(X, Y)
result_sbd_indep = sbd_indep(X, Y)
result_sbd_dep   = sbd_dep(X, Y)
result_dtw_indep = dtw_indep(X, Y)
result_dtw_dep   = dtw_dep(X, Y)

print("Euclidean Distance: ", result_euc)
print("SBD-Independent Distance: ", result_sbd_indep)
print("SBD-Dependent Distance: ", result_sbd_dep)
print("DTW-Independent Distance: ", result_dtw_indep)
print("DTW-Dependent Distance: ", result_dtw_dep)
```

## 5. Running clustering experiments
We provide an example of performing clustering experiments by running [src/Clustering/Clustering_pipeline.py](src/Clustering/Clustering_pipeline.py) script, with the following arguments:
- `-p` - path to the data directory
- `-f` - name of the dataset to run clustering on (e.g. BasicMotions)
- `-a` - name of the clustering algorithm to use (e.g. PAM_DTW_D)
- `-i` - the iteration index (e.g. 1)
- `-s` - path to save results

Example: Run clustering with PAM + DTW-D on the BasicMotions dataset with Nonorm we would run:
```shell
python ./src/Clustering/Clustering_pipeline.py -p $DATASET_DIR$ -f BasicMotions -a PAM_DTW_D -i 1 -s ./Clustering_results
```

## 6. Running anomaly detection experiments
We provide an example of performing anomaly detection experiments by running [src/Anomaly_Detection/AD_pipeline.py](src/Anomaly_Detection/AD_pipeline.py) script, with the following arguments:
- `-p` - path to the data directory
- `-f` - name of the dataset to run anomaly detection on (e.g. 010_MSL_id_9_Sensor_tr_554_1st_1172.csv)
- `-s` - path to save results
- `-sw` - the sliding window size (e.g. 100)
- `-ws` - the strategy to determine the sliding window size (options: auto, fixed)
- `-st` - the stride (e.g. 50)
- `-dm` - name of the measure to use (e.g. euclidean)

Example: Run anomaly detection (sliding window = 100 and stride = 50) with euclidean on the 010_MSL_id_9_Sensor_tr_554_1st_1172.csv with Nonorm we would run:
```shell
python ./src/Anomaly_Detection/AD_pipeline.py -p $DATASET_DIR$ -f 010_MSL_id_9_Sensor_tr_554_1st_1172.csv -s ./AD_results -sw 100 -ws fixed -st 50 -dm euclidean
```