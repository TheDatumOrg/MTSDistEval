import os
import pandas as pd
from scipy.stats import wilcoxon
# Surpress all warnings
import warnings
warnings.filterwarnings("ignore")

DATASETS = [
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
    "InsectWingbeat",
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
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
    "EigenWorms",
    "FaceDetection"
]

def load(input_dir) -> pd.DataFrame:
    df = pd.DataFrame(columns=["archive","measures","problem","RI"])
    # loop over measures
    for measure in os.listdir(input_dir):
        measure_dir = os.path.join(input_dir, measure)
        if not os.path.isdir(measure_dir):
            continue
        # loop over experiments (experiment_1, experiment_2, …)
        for exp in os.listdir(measure_dir):
            exp_dir = os.path.join(measure_dir, exp)
            if not os.path.isdir(exp_dir):
                continue
            # loop over datasets
            for dataset in os.listdir(exp_dir):
                eval_file = os.path.join(exp_dir, dataset, "evaluation_clustering.csv")
                if not os.path.exists(eval_file):
                    continue
                # read file
                tmp = pd.read_csv(eval_file)
                # take the first 3 columns (archive, measures, problem)
                archive = tmp.loc[0, "archive"]
                meas    = tmp.loc[0, "measures"]
                problem = tmp.loc[0, "problem"]
                # average RI over this run (in case file has multiple rows)
                avg_ri = tmp["RI"].mean()
                # add to df
                df.loc[len(df)] = [archive, meas, problem, avg_ri]
    # now average again across 10 runs
    df = df.groupby(["archive","measures","problem"], as_index=False)["RI"].mean()
    return df
# OK

def type_analysis(results, reference_dict, ref_name:tuple = ('ref, refnorm'), alpha=0.1) -> pd.DataFrame:
    """
    Compares the results of one list of measures with different normalizations to a reference config (e.g. L2 with zscore)

    Args:
    results: pd.DataFrame with columns: dataset, metric, normalization, accuracy
    reference_dict: dict with dataset as key and accuracy as value
    ref_name: tuple with the name of the reference config
    alpha: significance level for the wilcoxon signed-rank test

    returns:
    pd.DataFrame with columns: metric, normalization, accuracy, >, =, <, p_value, better
    """

    def wilcox_p(g):
        try:
            greater = wilcoxon(g.acc, g.ref, alternative='greater')[1] < alpha
            less = wilcoxon(g.acc, g.ref, alternative='less')[1] < alpha
            diff_p = wilcoxon(g.acc, g.ref, alternative='two-sided')[1]
            return pd.Series([greater, less, diff_p], index=['greater', 'less', 'diff_p'])
        except ValueError:
            return pd.Series([0, 0, 0], index=['greater', 'less', 'diff_p'])
    
    # Add reference to the lockstep table
    results['ref'] = results.problem.map(reference_dict)

    # Drop rows where the reference is missing
    results = results.dropna(subset=['ref'])

    ref_avg = results.ref.mean()

    # Get better/worse than L2
    results[">"] = results.acc > results.ref
    results["<"] = results.acc < results.ref
    results['='] = results.acc == results.ref

    # Get aggregates
    stats = results.groupby(["metric", "norm"]).agg({'acc': 'mean', '>': 'sum', '=': 'sum', '<': 'sum'}).reset_index()

    # Get the p-values for the wilcoxon signed-rank test
    stats[['greater', 'worse', 'diff_p']] = results.groupby(["metric", "norm"]).apply(wilcox_p).values

    # Sort on metric and norm
    stats = stats.sort_values(['metric', 'acc'], ascending=[True, False])

    # Add reference row to the bottom
    refrow = [ref_name[0], ref_name[1], ref_avg, 0, 0, 0, False, False, 1]
    stats.loc[len(stats)] = refrow

    return stats
# OK

def sliding(df):
    # filter out
    view = df[df.metric.isin(['PAM_SBD_D', 'PAM_Lorentzian', 'PAM_SBD_I', 'PAM_ED'])]
    # Set baseline
    PAM_ED = view[view.metric.isin(['PAM_ED'])]
    PAM_ED_ref = PAM_ED.set_index("problem").acc.to_dict()

    filtered_stats = type_analysis(view[view.metric != 'PAM_ED'].copy(), PAM_ED_ref, ref_name=('PAM_ED', 'nonorm'), alpha=.05)
    # Sort
    custom_order = ['PAM_SBD_D', 'PAM_SBD_I', 'PAM_Lorentzian', 'PAM_ED']
    filtered_stats['metric'] = pd.Categorical(filtered_stats['metric'], categories=custom_order, ordered=True)
    filtered_stats = filtered_stats.sort_values('metric')
    # Create table
    print("Table 9. Pairwise comparison of lock-step and sliding measures with Euclidean and elastic measures with SBD-D on the task of clustering on the UEA archive (Nonorm). See Table 3 for column descriptions.")
    print(filtered_stats)
    print("\n\n")
# OK

def elastic(df):
    # Filter out
    view = df[df.metric.isin(['PAM_MSM_D','PAM_MSM_I','PAM_LCSS_D','PAM_LCSS_I','PAM_ERP_D','PAM_ERP_I',\
                           'PAM_TWE_D','PAM_TWE_I', 'PAM_DTW_D','PAM_DTW_I', 'PAM_SBD_D'])]
    # Set baseline
    PAM_SBD_D = view[view.metric.isin(['PAM_SBD_D'])]
    PAM_SBD_D_ref = PAM_SBD_D.set_index("problem").acc.to_dict()

    filtered_stats = type_analysis(view[view.metric != 'PAM_SBD_D'].copy(), PAM_SBD_D_ref, ref_name=('PAM_SBD_D', 'nonorm'), alpha=.05)
    
    custom_order = ['PAM_DTW_D', 'PAM_DTW_I', 'PAM_ERP_I','PAM_MSM_D', 'PAM_MSM_I', 'PAM_ERP_D', 'PAM_TWE_I', 'PAM_TWE_D', 'PAM_LCSS_I', 'PAM_LCSS_D', 'PAM_SBD_D']
    filtered_stats['metric'] = pd.Categorical(filtered_stats['metric'], categories=custom_order, ordered=True)
    filtered_stats = filtered_stats.sort_values(by='metric')
    # Create table
    print("Table 2: Pairwise comparison of Elastic measures - Clustering")
    print(filtered_stats)
    print("\n\n")

def main(input_dir, output_dir):
    OUTDIR = output_dir
    os.makedirs(OUTDIR, exist_ok=True)

    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open(os.path.join(OUTDIR, 'tables.txt'), 'w') as f:
        sys.stdout = f
        # Load data
        df = load(input_dir=input_dir)
        df.rename(columns={"measures": "metric"}, inplace=True)
        df.rename(columns={"RI": "acc"}, inplace=True)
        df['norm'] = 'nonorm'
        sliding(df)
        elastic(df)

    sys.stdout = original_stdout

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for clustering experiments")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data directory")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    main(args.input, args.output)
