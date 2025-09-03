#!/usr/bin/env bash
set -euo pipefail

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

# Algorithms
ALGOS=(PAM_DTW_D PAM_DTW_I PAM_ED PAM_ERP_D PAM_ERP_I PAM_LCSS_D PAM_LCSS_I PAM_Lorentzian PAM_MSM_D PAM_MSM_I PAM_SBD_D PAM_SBD_I PAM_TWE_D PAM_TWE_I)

# Folders
FOLDERS=(
  ArticularyWordRecognition
  AtrialFibrillation
  BasicMotions
  CharacterTrajectories
  Cricket
  DuckDuckGeese
  Epilepsy
  EthanolConcentration
  ERing
  FingerMovements
  HandMovementDirection
  Handwriting
  Heartbeat
  JapaneseVowels
  Libras
  MotorImagery
  NATOPS
  PenDigits
  PEMS-SF
  PhonemeSpectra
  RacketSports
  SelfRegulationSCP1
  SelfRegulationSCP2
  SpokenArabicDigits
  StandWalkJump
  LSST
  UWaveGestureLibrary
  EigenWorms
  FaceDetection
  InsectWingbeat
)

# Run experiments: each algo × dataset × 10 runs
for i in $(seq 1 10); do
  for folder in "${FOLDERS[@]}"; do
    for algo in "${ALGOS[@]}"; do
      echo "[RUN] $algo | $folder | experiment-$i"
      python src/Clustering/Clustering_pipeline.py -p "$DATASET_PATH" -f "$folder" -a "$algo" -i "$i" -s "$OUTPUT_PATH"
    done
  done
done
