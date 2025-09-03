#!/bin/bash

# This script runs the full reproducibility study, including running of experiments and generating plots

# Path to the UEA archive
UEA_PATH=$1

# Path to the TSB-AD-M archive
TSB_PATH=$2

# Ask for dataset path if not given
if [ -z "$UEA_PATH" ]; then
  read -p "Enter UEA Archive path: " UEA_PATH
fi

# Ask for TSB-AD-M path if not given
if [ -z "$TSB_PATH" ]; then
  read -p "Enter TSB-AD-M path: " TSB_PATH
fi

# ----------------- RUNNING EXPERIMENTS ------------------------
CLS_OUTPUT=./output/classification
CLU_OUTPUT=./output/clustering
AD_OUTPUT=./output/anomaly_detection

echo "Running classification experiments"
bash scripts/classification/run_classification_exp.sh $UEA_PATH $CLS_OUTPUT

echo "Running clustering experiments"
bash scripts/clustering/run_clustering_exp.sh $UEA_PATH $CLU_OUTPUT

echo "Running anomaly detection experiments"
bash scripts/anomaly_detection/run_anomaly_exp.sh $TSB_PATH $AD_OUTPUT


# ----------------- GENERATING PLOTS ------------------------

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi

# Activate virtual environment
source .venv/bin/activate

python -m scripts.generate_plots --cls_results=$CLS_OUTPUT/inference/classification_results.csv --clu_results=$CLU_OUTPUT --ad_results=$AD_OUTPUT --output=./plots
