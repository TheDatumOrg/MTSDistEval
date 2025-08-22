# A Structured Study of Multivariate Time-Series Distance Measures

Main Recent Update:

+ [Feb, 2025] Paper accepted to ACM SIGMOD! The full paper can be found at [paper link](https://dl.acm.org/doi/10.1145/3725258).

Related Repository:

+ [TSDistEval](https://github.com/TheDatumOrg/TSDistEval): Debunking Four Long-Standing Misconceptions of Time-Series Distance Measures

If you find our work helpful, please consider citing:

<details>
<summary>"A Structured Study of Multivariate Time-Series Distance Measures" ACM SIGMOD 2025.</summary>

```bibtex
@inproceedings{dhondt2025structured,
  title={A Structured Study of Multivariate Time-Series Distance Measures},
  author={d’Hondt, Jens E and Li, Haojun and Yang, Fan and Papapetrou, Odysseas and Paparrizos, John},
  booktitle={Proceedings of the 2025 ACM SIGMOD international conference on management of data},
  year={2025}
}
```

</details>

<details>
<summary>"Debunking four long-standing misconceptions of time-series distance measures" ACM SIGMOD 2020.</summary>

```bibtex
@inproceedings{paparrizos2020debunking,
  title={Debunking four long-standing misconceptions of time-series distance measures},
  author={Paparrizos, John and Liu, Chunwei and Elmore, Aaron J and Franklin, Michael J},
  booktitle={Proceedings of the 2020 ACM SIGMOD international conference on management of data},
  pages={1887--1905},
  year={2020}
}
```

</details>

## Table of Contents

- [📄 Overview](#overview)
- [⚙️ Get Started](#start)

  * [💻 Installation](#installation)
  * [🗄️ Dataset](#dataset)
- [🏄‍♂️ Dive into MTS Distance Benchmark Study](#MTS)
- [✉️ Contact](#contact)
- [🎉 Acknowledgement](#ack)


<h2 id="overview"> 📄 Overview </h2>

Distance measures are fundamental to time series analysis and have been extensively studied for decades. Until now, research efforts mainly focused on univariate time series, leaving multivariate cases largely under-explored. Furthermore, the existing experimental studies on multivariate distances have critical limitations: (a) focusing only on lock-step and elastic measures while ignoring categories such as sliding and kernel measures; (b) considering only one normalization technique; and (c) placing limited focus on statistical analysis of findings. Motivated by these shortcomings, we present the most complete evaluation of multivariate distance measures to date. Our study examines 30 standalone measures across 8 categories, 2 channel-dependency models, and considers 13 normalizations. We perform a comprehensive evaluation across 30 datasets and 3 downstream tasks, accompanied by rigorous statistical analysis. To ensure fairness, we conduct a thorough investigation of parameters for methods in both a supervised and an unsupervised manner. Our work verifies and extends earlier findings, showing that insights from univariate distance measures also apply to the multivariate case: (a) alternative normalization methods outperform Z-score, and for the first time, we demonstrate statistical differences in certain categories for the multivariate case; (b) multiple lock-step measures are better suited than Euclidean distance, when it comes to multivariate time series; and (c) newer elastic measures outperform the widely adopted Dynamic Time Warping distance, especially with proper parameter tuning in the supervised setting. Moreover, our results reveal that (a) sliding measures offer the best trade-off between accuracy and runtime; (b) current normalization techniques fail to significantly enhance accuracy on multivariate time series and, surprisingly, do not outperform the no normalization case, indicating a lack of appropriate solutions for normalizing multivariate time series; and (c) independent consideration of time series channels is beneficial only for elastic measures. In summary, we offer guidelines to aid in designing and selecting preprocessing strategies and multivariate distance measures for our community.

<h2 id="start"> ⚙️ Get Started </h2>

<h3 id="installation">💻 Installation</h3>

Set up the environment with required packages:

```shell
pip install -r requirements.txt
```

<h3 id="dataset">🗄️ Dataset</h3>

Getting the data from:

- The original UEA archive, preprocessed to have equal-length time-series can be downloaded from the following [link](https://drive.google.com/file/d/1XWgiN0-1K3HaC4CWHvQjTHVdVhajMzH_/view?usp=sharing).
- The downsampled version of the UEA archive (used to evaluate elastic measures) can be downloaded from [here](https://drive.google.com/file/d/1gYPohWLcpYWuTf9mTdyDl0TfH2aXyfV5/view?usp=drive_link).
- The TSB-AD-M archive can be downloaded from [here](https://www.thedatum.org/datasets/TSB-AD-M.zip), with the related repository available [here](https://github.com/TheDatumOrg/TSB-AD).

<h2 id="MTS"> 🏄‍♂️ Dive into MTS Distance Benchmark Study </h2>

### 1. Example Usage

This repository primarily focuses on benchmarking experiments. To compute MTS distances using this code, please refer to the example as bellow. A complete list of supported MTS distances is available in `src/onennclassifier.py`.

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

### 2. Replicating the Results:

To replicate the results from our paper:

1. Run all experiments:
```shell
bash scripts/run_classification_exp.sh
```

2. Generate all tables and plots from the paper:
```shell
python generate_plots.py
```

> **Note**: This is a large-scale evaluation study. The classification experiments alone include over 18,000 runs, with individual runs taking anywhere from a few seconds to several hours. While the original evaluation was performed on a high-performance cluster with 100+ nodes running in parallel, the provided script runs experiments sequentially to accommodate users without access to such hardware resources.

### 3. Running an individual classification experiment.

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

### 4. Running an individual clustering experiment.

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

### 5. Running an individual anomaly detection experiment.

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

<h2 id="contact"> ✉️ Contact </h2>

If you have any questions or suggestions, feel free to contact:
* Jens E. d'Hondt (j.e.d.hondt@tue.nl)
* Haojun Li (li.14118@osu.edu)
* Fan Yang (yang.7007@osu.edu)
* Odysseas Papapetrou (o.papapetrou@tue.nl)
* John Paparrizos (paparrizos.1@osu.edu)

Or describe it in Issues.

<h2 id="ack"> 🎉 Acknowledgement </h2>

We would like to acknowledge Ryan DeMilt (demilt.4@osu.edu) for the valuable contributions to this work. Also appreciate the following github repos a lot for their valuable code base:
* https://github.com/TheDatumOrg/TSDistEval
* https://github.com/sktime/sktime
* https://github.com/aeon-toolkit/aeon
* https://github.com/dotnet54/multivariate-measures




