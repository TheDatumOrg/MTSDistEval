#!/bin/bash

DATASET_PATH=$1
OUTPUT_PATH=$2

# Ask for dataset path if not given
if [ -z "$DATASET_PATH" ]; then
  read -p "Enter dataset path: " DATASET_PATH
fi

# Ask for output path if not given
if [ -z "$OUTPUT_PATH" ]; then
  read -p "Enter output path: " OUTPUT_PATH
fi

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi

# Activate virtual environment
source .venv/bin/activate

# Set important variables
PARAM_PATH=./resources/validation_results.csv

PMEASURES=(dtw-d dtw-i lcss-d lcss-i erp-d erp-i twe-d twe-i msm-d msm-i rbf gak-d gak-i sink-d-denom sink-i-denom kdtw-d kdtw-i pca hmm-rescale-d hmm-rescale-i grail-d-denom grail-i-denom ts2vec-d ts2vec-i tloss-d tloss-i)
NPMEASURES=(l2 l1 lorentzian avg_l1_inf jaccard emanon4 soergel topsoe clark chord canberra kl-d kl-i eros sbd-i sbd-d catch22-i tsfresh-i)
ENSEMBLES=('sbd-d&dtw-i' 'sbd-d&msm-i')

NORMS=(zscore-i zscore-d minmax-i minmax-d median-i median-d mean-i mean-d unit-i unit-d sigmoid-i tanh-i adaptive none)

# Run experiments with optimal parameters on nonorm
for measure in "${PMEASURES[@]}"; do
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -pp $PARAM_PATH -m $measure -p '*' -n none
done
for measure in "${ENSEMBLES[@]}"; do
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -pp $PARAM_PATH -m $measure -p '*' -n none
done

# Run experiments with unsupervised parameters on all normalizations and datasets
for norm in "${NORMS[@]}"; do
    # Measures without parameters
    for measure in "${NPMEASURES[@]}"; do
        python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m $measure -p '*' -n $norm
    done
    # Measures with parameters
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m dtw-d -p '*' -n $norm -c sakoe_chiba_radius=0.1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m dtw-d -p '*' -n $norm -c sakoe_chiba_radius=1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m dtw-i -p '*' -n $norm -c sakoe_chiba_radius=0.1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m dtw-i -p '*' -n $norm -c sakoe_chiba_radius=1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m erp-d -p '*' -n $norm -c sakoe_chiba_radius=1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m gak-d -p '*' -n $norm -c sigma=0.1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m gak-i -p '*' -n $norm -c sigma=0.1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m kdtw-d -p '*' -n $norm -c sigma=0.125
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m kdtw-i -p '*' -n $norm -c sigma=0.125
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m lcss-d -p '*' -n $norm -c sakoe_chiba_radius=0.05 epsilon=0.2
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m lcss-i -p '*' -n $norm -c sakoe_chiba_radius=0.05 epsilon=0.2
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m msm-d -p '*' -n $norm -c c=0.5
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m msm-i -p '*' -n $norm -c c=0.5
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m rbf -p '*' -n $norm -c gamma=-1
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m sink-d-denom -p '*' -n $norm -c gamma=5
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m sink-i-denom -p '*' -n $norm -c gamma=5
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m twe-d -p '*' -n $norm -c lmbda=1 nu=0.0001
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m twe-i -p '*' -n $norm -c lmbda=1 nu=0.0001
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m tloss-d -p '*' -n $norm -c out_channels=320
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m tloss-i -p '*' -n $norm -c out_channels=320
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m ts2vec-d -p '*' -n $norm -c out_channels=320
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m ts2vec-i -p '*' -n $norm -c out_channels=320
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m pca -p '*' -n $norm -c exvar=.95
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m hmm-rescale-d -p '*' -n $norm -c n_states=2
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m hmm-rescale-i -p '*' -n $norm -c n_states=2
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m grail-d-denom -p '*' -n $norm -c gamma=2
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m grail-i-denom -p '*' -n $norm -c gamma=2
    python -m src.classification -mp inference -d $DATASET_PATH -o $OUTPUT_PATH -m erp-i -p '*' -n $norm -c sakoe_chiba_radius=1
done