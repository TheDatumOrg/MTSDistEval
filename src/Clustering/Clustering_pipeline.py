import os
import argparse
import numpy as np
import kmedoids
from k_Medoids_Elastic.dtw import dtw_all
from k_Medoids_Elastic.lcss import lcss_all
from k_Medoids_Elastic.erp import erp_all
from k_Medoids_Elastic.twe import twe_all
from k_Medoids_Elastic.msm import msm_all
from k_Medoids_SBD.sbd_numba_rocket_test import SBD_Local_all_rocket, SBD_Global_all_rocket
from k_Medoids_Lockstep.lp import euclidean_all, lorentzian_all
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import rand_score
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=False, default="UEA_downsampled")
    parser.add_argument('-f','--folder', required=True)
    parser.add_argument('-a', '--algo', required=False, default='PAM_DTW_D')
    parser.add_argument('-i', '--itr', required=False, default=1)
    parser.add_argument('-s','--save_path', required=True)
    parser.add_argument('-t','--testrun', action="store_true", help="Run in test mode if this flag is set")
    arguments = parser.parse_args()
    PATH = arguments.path
    folder = arguments.folder
    algo = arguments.algo
    itr = int(arguments.itr)
    save_path = arguments.save_path
    testrun = arguments.testrun

    if testrun: # Generate dummy data
          print("Running in test mode with random data.")
          X_train = np.random.randn(10,3,32)
          Y_train = np.random.randint(0, 2, size=(10,))
          X_test = np.random.randn(10,3,32)
          Y_test = np.random.randint(0, 2, size=(10,))
    else:
      X_train  = np.load(os.path.join(PATH, folder, f'{folder}_train_X.npy'))
      X_test  = np.load(os.path.join(PATH, folder, f'{folder}_test_X.npy'))
      Y_train  = np.load(os.path.join(PATH, folder, f'{folder}_train_Y.npy'))
      Y_test  = np.load(os.path.join(PATH, folder, f'{folder}_test_Y.npy'))
    
    label_encode = LabelEncoder()
    Y_train_norm = label_encode.fit_transform(Y_train)
    Y_test_norm = label_encode.transform(Y_test)

    ts = np.concatenate((X_train, X_test), axis=0)
    labels = np.append(Y_train_norm, Y_test_norm)

    num_clusters = len(set(labels))

    if testrun: # Generate random distances
       dist_mat = np.abs(np.random.rand(len(ts), len(ts)))
    elif algo == "PAM_DTW_D":
      dist_mat = dtw_all(ts, ts, mode='dependent', sakoe_chiba_radius=None, itakura_max_slope=None)
    elif algo == "PAM_DTW_I":
      dist_mat = dtw_all(ts, ts, mode='independent', sakoe_chiba_radius=None, itakura_max_slope=None)
    elif algo == "PAM_LCSS_D":
      dist_mat = lcss_all(ts, ts, mode='dependent', epsilon=0.5, sakoe_chiba_radius=0.1)
    elif algo == "PAM_LCSS_I":
      dist_mat = lcss_all(ts, ts, mode='independent', epsilon=1.0, sakoe_chiba_radius=0.05)
    elif algo == "PAM_ERP_D":
      dist_mat = erp_all(ts, ts, mode='dependent')
    elif algo == "PAM_ERP_I":
      dist_mat = erp_all(ts, ts, mode='independent')
    elif algo == "PAM_TWE_D":
      dist_mat = twe_all(ts, ts, mode='dependent', lmbda=1.0, nu=0.0001)
    elif algo == "PAM_TWE_I":
      dist_mat = twe_all(ts, ts, mode='independent', lmbda=0.5, nu=0.01)
    elif algo == "PAM_MSM_D":
      dist_mat = msm_all(ts, ts, mode='dependent', c=0.5)
    elif algo == "PAM_MSM_I":
      dist_mat = msm_all(ts, ts, mode='independent', c=0.5)
    elif algo == "PAM_SBD_D":
      dist_mat = SBD_Global_all_rocket(ts, ts)
    elif algo == "PAM_SBD_I":
      dist_mat = SBD_Local_all_rocket(ts, ts)
    elif algo == "PAM_Lorentzian":
      dist_mat = lorentzian_all(ts, ts)
    else:
      dist_mat = euclidean_all(ts, ts)
    # Store results
    results = kmedoids.pam(dist_mat, num_clusters, max_iter=100, init='random', random_state=None)
    predictions = results.labels
    dir_path = os.path.join(save_path, algo, f'experiment-{itr}', folder)
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    
    # Save the array to the file
    df = pd.DataFrame(data={'archive':[PATH], 'measures':[algo], 'problem': [folder], \
                            'RI': [rand_score(labels, predictions)]})
    csv_path = os.path.join(dir_path, 'evaluation_clustering.csv')
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
# OK