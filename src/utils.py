import os
import multiprocessing

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def initialise_multithread(num_cores=-1):
    """
    Initialise pool workers for multi processing
    :param num_cores:
    :return:
    """
    if (num_cores == -1) or (num_cores >= multiprocessing.cpu_count()):
        num_cores = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(num_cores)
    return p


def create_directory(directory_path):
    """
    Create a directory if path doesn't exists
    :param directory_path:
    :return:
    """
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path
    
def compute_classification_metrics(y_true,y_pred,y_true_val=None,y_pred_val=None):
    res = {}
    res['acc'] = accuracy_score(y_true,y_pred)
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['f1'] = f1_score(y_true,y_pred,average='macro')
    
    return res

multivariate = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "LSST",
    "UWaveGestureLibrary",
    "EigenWorms",
    "FaceDetection",
    "InsectWingbeat",
]

dataset_1 = [
   "PhonemeSpectra"
]
dataset_2 = [
   "RacketSports",
   "SelfRegulationSCP1",
   "SelfRegulationSCP2"
]
dataset_3 = [
    "StandWalkJump",
    "UWaveGestureLibrary",
    "Heartbeat",
    "FingerMovements"
]
dataset_4 = [
    "HandMovementDirection",
    "Handwriting",
    "MotorImagery",
    "SpokenArabicDigits"
]

dataset_5 = [

    "EigenWorms",
    "FaceDetection",
    "InsectWingbeat",

]

dataset_6 = [
   "FaceDetection",
    "InsectWingbeat",
]

dataset_A = [
    "AtrialFibrillation",
    "BasicMotions",
    "ArticularyWordRecognition",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
    "SpokenArabicDigits"

]

dataset_B = [

    "FaceDetection",
    "EigenWorms",
    "InsectWingbeat"

]

# Check whether all rows of a matrix is znormalized
def is_row_z_normalized(matrix, tolerance=1e-10):
    # Initialize an empty list to store the check results
    row_checks = []
    # Iterate over each row
    for row in matrix:
        # Calculate mean and standard deviation of the row
        mean = np.mean(row)
        std_dev = np.std(row)
        if np.abs(std_dev) < tolerance: std_dev = 1
        # Check if the mean is approximately 0 and the standard deviation is approximately 1
        is_normalized = (np.abs(mean) < tolerance) and (np.abs(std_dev - 1) < tolerance)
        # Append the result to the list
        row_checks.append(is_normalized)
    return np.sum(np.array(row_checks))

# check whether the input tensor is z-normalized
def is_total_z_normalized(tensor, tolerance=1e-10):
  m = tensor.shape[0]
  d = tensor.shape[1]
  count = 0
  for i in range(m):
    if is_row_z_normalized(tensor[i],tolerance) == d:
      count += 1
    else:
      pass
  if count == m:
    return True
  else:
    return False
    

# TEST CASE
if __name__ == "__main__":

    a = set(multivariate)
    b = set(dataset_A).union(set(dataset_B))

    assert a == b
    print(len(a), len(b))
