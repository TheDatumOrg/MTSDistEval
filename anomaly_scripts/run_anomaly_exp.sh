#!/bin/bash

DATASET_PATH=$1

# Ask for dataset path if not given
if [ -z "$DATASET_PATH" ]; then
  read -p "Enter dataset path: " DATASET_PATH
fi

# If running inside a conda environment, just use it and install requirements
if [ -n "$CONDA_PREFIX" ]; then
  echo "Detected conda environment: $CONDA_DEFAULT_ENV"
  echo "Using current conda env"
#   pip install -r requirements.txt
  echo "Requirements installed"
else
  # Create virtual environment if not exists
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
  else
    # Activate virtual environment
    source .venv/bin/activate
  fi
fi

# set up params
RESULT_DIR="./AD_results"          # -s (change as needed)
SLIDE_WIN=100                      # -sw
WIN_STRAT="fixed"                  # -ws
STEP=50                            # -st

# Candidate models for -dm
MODELS=("Lorentzian" "euclidean" "SBD_D" "SBD_I" "DTW_D" "DTW_I")

# Create results dir
mkdir -p "$RESULT_DIR"

echo "Dataset path: $DATASET_PATH"
echo "Saving results to: $RESULT_DIR"
echo "Models: ${MODELS[*]}"

# --------- LOOPS ----------
# Find all .csv files under DATASET_PATH (recursively), robust to spaces/newlines
while IFS= read -r -d '' csv_path; do
  csv_file="$(basename "$csv_path")"   # AD_pipeline.py expects filename for -f
  for model in "${MODELS[@]}"; do
    echo ">>> Running: -f $csv_file  -dm $model"
    python ./src/Anomaly_Detection/AD_pipeline.py \
      -p "$DATASET_PATH" \
      -f "$csv_file" \
      -s "$RESULT_DIR" \
      -sw "$SLIDE_WIN" \
      -ws "$WIN_STRAT" \
      -st "$STEP" \
      -dm "$model" 

  done
done < <(find "$DATASET_PATH" -type f -name '*.csv' -print0)


