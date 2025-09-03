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

# set up params
SLIDE_WIN=100                      # -sw
WIN_STRAT="fixed"                  # -ws
STEP=50                            # -st

# Candidate models for -dm
MODELS=("Lorentzian" "euclidean" "SBD_D" "SBD_I" "DTW_D" "DTW_I")

# Create results dir
mkdir -p "$OUTPUT_PATH"

echo "Dataset path: $DATASET_PATH"
echo "Saving results to: $OUTPUT_PATH"
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
      -s "$OUTPUT_PATH" \
      -sw "$SLIDE_WIN" \
      -ws "$WIN_STRAT" \
      -st "$STEP" \
      -dm "$model"

  done
done < <(find "$DATASET_PATH" -type f -name '*.csv' -print0)


