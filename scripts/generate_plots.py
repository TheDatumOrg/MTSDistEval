import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

METRIC_MAP = {
        "avg_l1_inf": r"$L_{1,avg,\infty}$",
        "canberra": "Canberra",
        "catch22-i": "Catch22-I",
        "catch22-d": "Catch22-D",
        "chord": "Chord",
        "clark": "Clark",
        "dtw-d": "DTW-D",
        "dtw-i": "DTW-I",
        "emanon4": "Emanon4",
        "eros": "EROS",
        "erp-d": "ERP-D",
        "erp-i": "ERP-I",
        "euclidean": r"$L_2$",
        "l2": r"$L_2$",
        "gak-d": "GAK-D",
        "gak-i": "GAK-I",
        "kdtw-d": "KDTW-D",
        "kdtw-i": "KDTW-I",
        "grail-d": "GRAIL-D",
        "grail-i": "GRAIL-I",
        "grail-d-denom": "GRAIL-D",
        "grail-i-denom": "GRAIL-I",
        "hmm-rescale-i": r"$KL_{HMM}$-I",
        "hmm-rescale-d": r"$KL_{HMM}$-D",
        "jaccard": "Jaccard",
        "kl-i": r"$KL_{Gauss}$-I",
        "kl-d": r"$KL_{Gauss}$-D",
        "l1": r"$L_1$",
        "l2": r"$L_2$",
        "lcss-d": "LCSS-D",
        "lcss-i": "LCSS-I",
        "lorentzian": "Lorentzian",
        "msm-d": "MSM-D",
        "msm-i": "MSM-I",
        "pca": "PCA",
        "rbf": "RBF",
        "sbd-d": "SBD-D",
        "sbd-i": "SBD-I",
        "sink-d": "SINK-D",
        "sink-d-denom": "SINK-D",
        "sink-i": "SINK-I",
        "sink-i-denom": "SINK-I",
        "soergel": "Soergel",
        "topsoe": "Topsoe",
        "tsfresh-i": "TSFresh-I",
        "tsfresh-d": "TSFresh-D",
        "twe-d": "TWE-D",
        "twe-i": "TWE-I",
        "tloss-d": "T-Loss-D",
        "tloss-i": "T-Loss-I",
        "ts2vec-d": "TS2Vec-D",
        "ts2vec-i": "TS2Vec-I",
}

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)
NORM_ORDER = ["z-score", "nonorm", "minmax", "tanh", "unit", "sigmoid", "mean", "adaptive", "median"]
PARAMLESS_MEASURES = ["l2", "l1", "lorentzian", "avg_l1_inf", "jaccard", "emanon4", "soergel", "topsoe", "clark", "chord", "canberra", "kl-d", "kl-i", "eros", "sbd-i", "sbd-d", "catch22-i", "tsfresh-i"]

def load() -> pd.DataFrame:
    df = pd.read_csv("inference/results/classification_results.csv")

    # Drop duplicates based on every column except runtime
    cols = df.columns.difference(['runtime'])
    df = df.drop_duplicates(subset=cols)

    return df

import json
import re

def preprocess(df):
    # Pre-processing
    def formal(m):
        if '&' in m:
            return " + ".join([formal(x) for x in m.split("&")])
        return METRIC_MAP.get(m, m)
    
    # LOOCV policy
    df.loc[df.parameters_given == False, 'metric_params'] = 'LOOCV'

    # Create formal name for metric
    df['metric_formal'] = df.metric.apply(formal)

    # Rename nonorm
    df.loc[df.norm == 'none', 'norm'] = 'nonorm'

    # Create formal name for normalization
    df['norm_formal'] = df.norm.str.capitalize()

    # Create category column
    with open('resources/categories.json') as f:
        CATEGORY_MAP = json.load(f)['categories']

    df['category'] = np.where(df.metric.str.contains("&"), "Ensemble", df.metric.apply(lambda x: CATEGORY_MAP.get(x.split("-")[0], np.nan)))

    # Create base measure column
    df['metric_base'] = df.metric.apply(lambda x: re.sub(r'-(global)$|-(local)$|-(d-denom)$|-(i-denom)$|-[id]{1}$', '', x))

    # Create a channel dependency column
    df['dependency'] = df.metric.apply(lambda x: "Dependent" if x.endswith("-d") or x.endswith("-d-denom") or x.endswith("-global") or x.endswith('-global-denom') else "Independent")

    return df

from scipy.stats import friedmanchisquare, wilcoxon
from scikit_posthocs import posthoc_nemenyi_friedman
from Friedman_Nemenyi_test import graph_ranks

def friedman_nemenyi(df, treatment, dataset, acc, alpha=.05):
    '''
    Expected input:
    df with columns: treatment, dataset, acc
    Example:
    0   eros_zscore ArticularyWordRecognition 0.5
    1   eros_zscore BasicMotions 0.7
    ...
    or 
    0   internal ArticularyWordRecognition 0.5
    1   internal BasicMotions 0.7
    '''
    # Reshape to wide format again
    pv = df.pivot_table(index=[treatment], columns=[dataset], values=acc)

    # Keep only the measures that have all datasets
    # pv = pv.dropna(axis=0, how="any")
    
    # Keep only the datasets that have all measures
    pv = pv.dropna(axis=1, how="any")

    # If the number of treatments is less than 3, we do the wilcoxon signed-rank test
    if pv.shape[0] < 3:
        pval = wilcoxon(*pv.values)[1]
        # print("P-value for wilcoxon test: ", pval)
    else:
        # Compute friedman p-value
        pval = friedmanchisquare(*pv.values)[1]
        # print("P-value for friedman test: ", pval)

    # If significant, do nemenyi test
    if pval < alpha:
        print("Significant difference between treatments, doing Nemenyi test")
        nemenyi = posthoc_nemenyi_friedman(pv.T.values).to_numpy()
    else:
        # Create dummydf = df.loc[df['runtime'].idxmax()] nemenyi with high p-values
        nemenyi = np.ones((pv.shape[0], pv.shape[0])) * pval
        
    ndf = pd.DataFrame(nemenyi, index=pv.index, columns=pv.index)
    # Get upper triangular of ndf as long df
    ndf = ndf.where(np.triu(np.ones(ndf.shape), k=1).astype(bool)).stack().rename_axis(["treatment1", "treatment2"]).reset_index(name="p_value")
    
    # Get the significant differences
    ndf.p_value = ndf.p_value < alpha


    return ndf, pv

def stat_test(view: pd.DataFrame, treatment, datasets, acc, alpha=0.05, 
              rank_threshold=None,
              super_treatment: str=None,
              outname=None, height=3, width=9, print=False, title=None, ax=None, 
              **kwargs):
    """
    Method that performs the Friedman and Nemenyi test on the given data to determine if there are significant differences between the treatments.
    Also outputs a critical difference diagram.

    args:
    view: DataFrame with the data in long format, with columns: treatment, dataset, acc
    treatment: name of the column with the treatments
    datasets: name of the column with the datasets
    acc: name of the column with the accuracies
    alpha: significance level for the tests
    outname: name of the output file for the critical difference diagram
    rank_threshold: threshold for the average ranks, only treatments with average ranks below this threshold will be considered
    """

    # Friedman and Nemenyi test
    p_vals, pv = friedman_nemenyi(view, treatment, datasets, acc, alpha=alpha)


    # Get average ranks per treatment
    if super_treatment is not None:
        avg_ranks_super = view.groupby([super_treatment, treatment, datasets]).acc.mean().unstack().rank(ascending=False, axis=0).mean(axis=1).reset_index()
        avg_ranks = avg_ranks_super.groupby(treatment)[0].mean().sort_values(ascending=False)
    else:
        avg_ranks = pv.rank(ascending=False, axis=0).mean(axis=1).sort_values(ascending=False)

    if print:
        print("Average ranks:", avg_ranks, "\n")

    if rank_threshold is not None:
        avg_ranks = avg_ranks[avg_ranks <= rank_threshold]
        # Make sure p_vals are also filtered
        p_vals = p_vals[p_vals.treatment1.isin(avg_ranks.index) & p_vals.treatment2.isin(avg_ranks.index)]

    # Prepare p_values for graphing
    graph_pvals = p_vals.values
    # Add dummy 3rd column
    graph_pvals = np.c_[graph_pvals[:, :-1], np.zeros(graph_pvals.shape[0]), graph_pvals[:, -1]]

    # Plot the results
    graph_ranks(avg_ranks.values, avg_ranks.keys(), graph_pvals,
                    cd=None, reverse=True,
                    width=width,
                    height=height, ax=ax, **kwargs)
    
    if title is not None:
        if ax is None:
            plt.title(title, y=kwargs.get("titley", 0.75), fontsize=kwargs.get("titlesize", 15))
        else:
            ax.set_title(title, y=kwargs.get("titley", 0.75), fontsize=kwargs.get("titlesize", 15))
    
    if outname is not None and ax is None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

    return p_vals, avg_ranks

from scipy.stats import wilcoxon

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
            return 1
    
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

def wilcox_p(g, acc='acc', alpha=0.1):
    gleft = g[f"{acc}_left"]
    gright = g[f"{acc}_right"]

    nworse = (gleft > gright).sum()
    nequal = (gleft == gright).sum()
    nbetter = (gleft < gright).sum()

    if wilcoxon(gleft, gright, alternative='greater')[1] < alpha: # Worse than baseline
        wilx = -1
    elif wilcoxon(gleft, gright, alternative='less')[1] < alpha: # Better than baseline
        wilx = 1
    else:
        wilx = 0

    return pd.Series([nbetter, nequal, nworse, wilx], index=['>', '=', '<', 'p_value'])

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('pdf', fonttype=42)

DATASET_MAPPING = {
    "ArticularyWordRecognition": "AWR",
    "AtrialFibrillation": "AF",
    "BasicMotions": "BM",
    "CharacterTrajectories": "CT",
    "Cricket": "CR",
    "DuckDuckGeese": "DDG", 
    "Epilepsy": "EP",
    "EthanolConcentration": "EC",
    "ERing": "ER",
    "FingerMovements": "FM",
    "HandMovementDirection": "HMD", 
    "Handwriting": "HW",
    "Heartbeat": "HB",
    "JapaneseVowels": "JV",
    "Libras": "LB",
    "MotorImagery": "MI", 
    "NATOPS": "NO",
    "PenDigits": "PD",
    "PEMS-SF": "PSF",
    "PhonemeSpectra": "PS",
    "RacketSports": "RS", 
    "SelfRegulationSCP1": "SR1", 
    "SelfRegulationSCP2": "SR2",
    "SpokenArabicDigits": "SAD",
    "StandWalkJump": "SWJ",
    "LSST": "LSST",
    "UWaveGestureLibrary": "UWGL",
    "EigenWorms": "EW",
    "FaceDetection": "FD",
    "InsectWingbeat": "IW",
}

def get_comparison_table(df):
    df['problem_base'] = df.problem + '_' + df.metric_base

    # Join the dataset with itself where metric_base and dataset is the same, but dependency is different
    view = df.groupby(['problem_base', 'problem', 'dependency']).acc.mean().unstack().reset_index()

    # # Compare the accuracies
    view['better'] = view.Dependent > view.Independent
    view['equal'] = view.Dependent == view.Independent
    view['worse'] = view.Dependent < view.Independent

    return view

from scipy.stats import wilcoxon
def get_wilcoxon_test(g):
    return wilcoxon(g.Dependent, g.Independent)[1]

# Group by dataset to see how many times dependent is better than independent
def histplot(view, title=None, ax=None, alpha=0.1, outname=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    # Map the dataset names
    view['problem'] = view.problem.map(DATASET_MAPPING)

    colors = sns.color_palette("colorblind")

    # Get the counts
    stats = view.groupby('problem')[['better', 'equal', 'worse']].sum().sort_values(['better', 'equal', 'worse'], ascending=False)

    # Plot the counts
    stats.plot(kind='bar', ax=ax, stacked=True, color=colors, width=0.8)

    # Do wilcoxon test on each dataset
    with_stars = False
    try:
        wilcox = view.groupby('problem').apply(get_wilcoxon_test)[stats.index]

        # Add a star above the datasets where wilcox test is significant
        stars_idx = np.where(wilcox < alpha)[0]

        if len(stars_idx) > 0:
            stars_y = stats.iloc[stars_idx].sum(axis=1) * 1.05
            ax.scatter(stars_idx, stars_y, marker='*', color='orange', s=300, label='significant')
            with_stars = True

    except ValueError:
        pass

    handles, labels = ax.get_legend_handles_labels()
    if with_stars:
        # Change the legend names
        ax.legend(handles, ['Significant', 'Dep. > Indep.', 'Dep == Indep.', 'Dep < Indep.'], fontsize=22, loc='lower left')
    else:
        ax.legend(handles, ['Dep. > Indep.', 'Dep. == Indep.', 'Dep < Indep.'], fontsize=22, loc='lower left')


    # ax.set_xlabel('Dataset', fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel("Frequency", fontsize=30)

    # Increase fontsize of y-tick labels
    # Rotate the x-axis labels
    ticklabelfontsize = 20
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=ticklabelfontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticklabelfontsize)

    if title:
        ax.set_title(title, fontsize='large', fontweight='bold')

    if outname:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

    return ax

import seaborn as sns
import matplotlib.pyplot as plt

def acc_to_runtime_plot(df, outname=None):
    fig, ax = plt.subplots(figsize=(12, 4.5))

    colors = sns.color_palette("tab10", n_colors=7)

    df.sort_values('category', inplace=True, ascending=False)

    ax = sns.scatterplot(data=df, x='runtime', y='mean_acc', hue='category', style='category', markers=['o', 'v', '^', '<', '>', 's', 'D'], ax=ax, s=200, palette=colors)

    ax.set_xscale('log')

    # Set all fontsizes to x
    fontsize = 18
    plt.rcParams.update({'font.size': fontsize})

    # Annotate the names of the metrics
    for i, row in df.iterrows():
        if row.metric_formal == 'MSM-I':
            ax.annotate(row['metric_formal'], (row['runtime'], row['mean_acc']), textcoords="offset points", xytext=(0,25), ha='center')
        elif row.metric_formal == 'DTW-I':
            ax.annotate(row['metric_formal'], (row['runtime'], row['mean_acc']), textcoords="offset points", xytext=(-5,10), ha='center')
        elif row.metric_formal in ['Euclidean', 'MSM-D', 'SBD-I', 'GAK-D']:
            ax.annotate(row['metric_formal'], (row['runtime'], row['mean_acc']), textcoords="offset points", xytext=(0,-20), ha='center')
        elif row.metric_formal == 'SINK-D':
            ax.annotate(row['metric_formal'], (row['runtime'], row['mean_acc']), textcoords="offset points", xytext=(5,10), ha='center')
        elif row.metric_formal == 'GAK-D':
            ax.annotate(row['metric_formal'], (row['runtime'], row['mean_acc']), textcoords="offset points", xytext=(-10,10), ha='center')
        elif row.metric_formal == 'GRAIL-D':
            ax.annotate(row['metric_formal'], (row['runtime'], row['mean_acc']), textcoords="offset points", xytext=(50,-10), ha='center')
        else:
            ax.annotate(row['metric_formal'], (row['runtime'], row['mean_acc']), textcoords="offset points", xytext=(0,10), ha='center')

    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0] - 0.05, ylims[1] + 0.07)

    xlims = ax.get_xlim()
    ax.set_xlim(xlims[0] / 5, xlims[1] * 5)


    # Draw the pareto frontier (Euclidean, SBD-D, MSM-I)
    l2 = df[df.metric == 'l2']
    lor = df[df.metric == 'lorentzian']
    sbd = df[df.metric == 'sbd-d']
    msm = df[df.metric == 'msm-i']

    points = [l2, lor, sbd, msm]

    xs = [row['runtime'].iloc[0] for row in points]
    ys = [row['mean_acc'].iloc[0] for row in points]

    # Add beginning and end
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xs.insert(0, xlims[0])
    ys.insert(0, ylims[0])
    xs.append(xlims[1])
    ys.append(msm['mean_acc'].iloc[0])

    ax.plot(xs, ys, 'k--', lw=2, label='Pareto frontier', zorder=0, color='gray')

    # Fill the area under the pareto frontier
    ax.fill_between(xs, ys, ylims[0], alpha=0.1, color='gray')
            
    # Remove title from legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title=None, ncol=2, fontsize=fontsize)

    # Set axis labels
    ax.set_xlabel('Inference runtime (in seconds) on AtrialFibrillation', fontsize=fontsize)
    ax.set_ylabel('Average accuracy on UEA', fontsize=fontsize)

    # Set font size of ticks
    ax.tick_params(axis='both', which='major', labelsize=fontsize)


    if outname:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

# =======================================================================

def norm_experiment(unsup):
    unsup['dataset_metric'] = unsup['problem'] + '_' + unsup['metric']

    # Global CD analysis
    _ = stat_test(unsup, treatment='norm_formal', datasets='dataset_metric', acc='acc', alpha=0.1, super_treatment='category', outname=os.path.join(OUTDIR, '4b-normalization_all.pdf'), 
                height=2.4, 
                title='Normalizations | All measures', titley=1,
                highlight=['Z-Score-I', 'Nonorm'], 
                fontsize=20, textspace=1, space_between_names=0.25,
                sign_line_height=0.075
                )

    # Category specific 
    for cat in ['Lock-step', 'Sliding', 'Elastic']:
        view = unsup[unsup.category == cat]

        _ = stat_test(view, treatment='norm_formal', datasets='dataset_metric', acc='acc', alpha=0.1, 
                    outname=os.path.join(OUTDIR, f'4b-normalization_{cat.lower()}.pdf'), 
                    height=2.4, 
                    title=f'Normalizations | {cat} measures', titley=1, 
                    highlight=['Z-Score-I', 'Nonorm'], 
                    sign_line_height=0.075,              
                    )
        
    # Meta ranking

    # Overall ranking
    overall_rank = unsup.groupby(['dataset_metric', 'norm']).acc.mean().unstack().rank(ascending=False, axis=1).mean(axis=0).sort_values().index.tolist()

    # Category specific
    cat_ranks = {}
    for cat in unsup.category.unique():
        cat_ranks[cat] = unsup[unsup.category == cat].groupby(['dataset_metric', 'norm']).acc.mean().unstack().rank(ascending=False, axis=1).mean(axis=0).sort_values().index.tolist()

    meta_rank = pd.DataFrame({
        'Overall': overall_rank,
        **cat_ranks
    })

    CAT_ORDER = ['Lock-step', 'Sliding', 'Elastic', 'Kernel', 'Feature-based', 'Model-based', 'Embedding', 'Overall']

    # Print to stdout
    print("Fig 4a; Meta ranking of normalization methods.")
    print(meta_rank[CAT_ORDER])
    print("\n\n")

def lockstep(df):
    view = df[(df.category == 'Lock-step') & (df.norm == 'nonorm')]

    print("Table 3: Pairwise comparison of Lock-step measures")
    stats = type_analysis(view[view.metric != 'l2'], l2_ref, ref_name=('l2', 'nonorm'), alpha=.05)
    print(stats)
    print("\n\n")

    focusview = view[view.metric.isin(['l2', 'lorentzian', 'avg_l1_inf'])]

    focusview.loc[focusview.metric == 'l2', 'metric_formal'] = 'Euclidean'

    _ = stat_test(focusview, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, outname=os.path.join(OUTDIR, '5a-lockstep_nonorm.pdf'), height=1.1, highlight='Euclidean', titley=.85, title='Lock-step vs. Euclidean')

def sliding(df):
    view = df[
    (df.norm == 'nonorm') &
    ((df.category == 'Sliding') | (df.metric == 'lorentzian'))
]

    stats = type_analysis(view[view.metric != 'lorentzian'], lor_ref, ref_name=('lorentzian', 'nonorm'), alpha=.05)

    print("Table 4: Pairwise comparison of Sliding measures")
    print(stats)
    print("\n\n")

    _ = stat_test(view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, outname=os.path.join(OUTDIR, '5b-sliding_nonorm.pdf'), height=1.1, highlight='Lorentzian', titley=.85, title='Sliding vs. Lorentzian')

def elastic(df):
    view = df[
    ((df.category == 'Elastic') | (df.metric == 'sbd-d')) &
    (df.norm == 'nonorm')
]

    # Pairwise comparison
    view_noref = view[view.metric != 'sbd-d']

    view_noref = view_noref.drop('norm', axis=1).rename(columns={'metric_params': 'norm'})

    stats = type_analysis(view_noref, sbd_ref, ref_name=('sbd-d', ''), alpha=0.05)

    print("Table 5: Pairwise comparison of Elastic measures")
    print(stats)
    print("\n\n")

    sup_view = view[
        (view.metric.isin(['sbd-d', 'msm-i', 'twe-i'])) &
        (view.metric_params == 'LOOCV')
    ]

    _ = stat_test(sup_view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, outname=os.path.join(OUTDIR, '5c-elastic_supervised.pdf'), height=1.1, highlight='SBD-D', titley=.85, title='Elastic vs. SBD-D (Supervised)')

    unsup_view = unsup[
        (unsup.norm == 'nonorm') &
        (unsup.metric.isin(['msm-i', 'twe-i', 'dtw-i-100', 'sbd-d']))
    ]

    unsup_view.loc[unsup_view.metric == 'dtw-i-100', 'metric_formal'] = 'DTW-I-100'

    _ = stat_test(unsup_view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, outname=os.path.join(OUTDIR, '5d-elastic_unsupervised.pdf'), height=1.1, highlight='SBD-D', titley=.85, title='Elastic vs. SBD-D (Unsupervised)')

def kernel(df):
    view = df[
        ((df.category == 'Kernel') | (df.metric == 'sbd-d')) &
        (df.norm == 'nonorm')
    ]

    view_noref = view[view.metric != 'sbd-d']

    view_noref = view_noref.drop('norm', axis=1).rename(columns={'metric_params': 'norm'})

    stats = type_analysis(view_noref, sbd_ref, ref_name=('sbd-d', ''), alpha=0.05)

    print("Table 6: Pairwise comparison of Kernel measures")
    print(stats)
    print("\n\n")

    sup_view = view[
        (view.metric.isin(['sbd-d', 'sink-d-denom', 'gak-d'])) &
        (view.metric_params == 'LOOCV')
    ]

    _ = stat_test(sup_view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, outname=os.path.join(OUTDIR, '6a-kernel_supervised.pdf'), height=1.1, highlight='SBD-D', titley=.85, title='Kernel vs. SBD-D (Supervised)')

    unsup_view = unsup[
        (unsup.norm == 'nonorm') &
        (unsup.metric.isin(['sink-d-denom', 'gak-d', 'sbd-d']))
    ]

    _ = stat_test(unsup_view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, outname=os.path.join(OUTDIR, '6b-kernel_unsupervised.pdf'), height=1.1, highlight='SBD-D', titley=.85, title='Kernel vs. SBD-D (Unsupervised)')

def feature(df):
    view = df[
        ((df.category == 'Feature-based') | (df.metric == 'l2')) &
        (df.norm == 'nonorm')
    ]

    stats = type_analysis(view[view.metric != 'l2'], l2_ref, ref_name=('l2', 'nonorm'), alpha=0.05)

    print("Table 7.1: Pairwise Comparison of Feature-based Methods")
    print(stats)
    print("\n\n")

    view.loc[view.metric == 'l2', 'metric_formal'] = 'Euclidean'

    _ = stat_test(view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, height=1.1, outname=os.path.join(OUTDIR, '7a-feature_based.pdf'), title='Feature-based vs. Euclidean', titley=0.85, highlight='Euclidean', fontsize=16, titlesize=18)

def model(df):
    view = df[
        ((df.category == 'Model-based') | (df.metric == 'l2')) &
        (df.norm == 'nonorm')
    ]

    stats = type_analysis(view[view.metric != 'l2'], l2_ref, ref_name=('l2', 'nonorm'), alpha=0.05)

    print("Table 7.2: Pairwise Comparison of Model-based Methods")
    print(stats)
    print("\n\n")

    view.loc[view.metric == 'l2', 'metric_formal'] = 'Euclidean'

    _ = stat_test(view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, height=1.4, outname=os.path.join(OUTDIR, '7b-model_based.pdf'), title='Model-based vs. Euclidean', titley=0.85, highlight='Euclidean', fontsize=16, titlesize=18)

def embedding(df):
    view = df[
        ((df.category == 'Embedding') | (df.metric == 'l2')) &
        (df.norm == 'nonorm')
    ]

    tmp = view.drop('norm', axis=1).rename(columns={'metric_params': 'norm'})

    stats = type_analysis(tmp[tmp.metric != 'l2'], l2_ref, ref_name=('Euclidean', ''), alpha=0.05)

    print("Table 7.3: Pairwise comparison of Embedding measures")
    print(stats)
    print("\n\n")

    sup_view = df[
        (df.norm == 'nonorm') &
        (df.metric.isin(
            ['grail-d-denom', 'grail-i-denom', 'ts2vec-d', 'ts2vec-i', 'tloss-d', 'tloss-i', 'l2', 'sbd-d']
            )) &
        (df.metric_params == 'LOOCV')
    ]

    _ = stat_test(sup_view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1, outname=os.path.join(OUTDIR, '7c-embedding_supervised.pdf'), height=2, highlight='SBD-D', titley=.95, title='Embeddings vs. SBD-D and Euclidean')

def ensemble(df):
    view = df[
        ((df.category == 'Ensemble') | (df.metric.isin(['dtw-i', 'msm-i', 'sbd-d']))) &
        (df.norm == 'nonorm')
    ]

    _ = stat_test(view, treatment='metric_formal', datasets='problem', acc='acc', alpha=0.1,height=1.7, title='Lockstep + Sliding Ensembles (Supervised)', highlight='SBD-D', titley=0.95, outname=os.path.join(OUTDIR, '7d-ensemble_lockstep_sliding_supervised.pdf'),fontsize=16, titlesize=18)

def dependency(df):
    channel_view = unsup[
        (unsup.norm == 'nonorm') &
        ~(unsup.metric.str.endswith('-100'))
    ]

    # Drop the measures for which we only have one dependency variant
    tmp = channel_view.groupby(['metric_base']).dependency.agg(['nunique', 'unique']).reset_index()

    channel_view = channel_view[channel_view.metric_base.isin(tmp[tmp["nunique"] == 2].metric_base.tolist())]

    view = get_comparison_table(channel_view)

    fig, ax = plt.subplots(figsize=(14, 5))

    histplot(view, title=None, alpha=0.1, outname=os.path.join(OUTDIR, '8-channels_all.pdf'), ax=ax)

def runtime(df):
    # Only get unsupervised times for runtime analysis (to avoid duplicates)
    view = unsup[
        (unsup.norm == 'nonorm') & 
        ~(unsup.metric.str.endswith('-100'))
    ]

    avg_acc = view.groupby(['metric', 'metric_formal', 'category']).acc.mean().reset_index()
    avg_acc.rename(columns={'acc': 'mean_acc'}, inplace=True)

    atrial = view[view.problem == 'AtrialFibrillation']

    # Get median runtime
    runtime = atrial.groupby(['metric', 'metric_formal', 'category']).runtime.median().reset_index()

    # Join with accuracies
    combined = pd.merge(avg_acc, runtime, on=['metric', 'metric_formal', 'category'])

    # Do the plot
    acc_to_runtime_plot(combined, outname=os.path.join(OUTDIR, '9a-runtime_vs_accuracy.pdf'))


if __name__ == "__main__":
    import sys

    # Redirect stdout to a file
    original_stdout = sys.stdout
    with open(os.path.join(OUTDIR, 'tables.txt'), 'w') as f:
        sys.stdout = f

        df = load()
        df = preprocess(df)

        # Get the baselines to be used in comparison
        l2_ref = df[(df.metric == 'l2') & (df.norm == 'nonorm')].set_index("problem").acc.to_dict()
        lor_ref = df[(df.metric == 'lorentzian') & (df.norm == 'nonorm')].set_index("problem").acc.to_dict()
        sbd_ref = df[(df.metric == 'sbd-d') & (df.norm == 'nonorm')].set_index("problem").acc.to_dict()
        
        # Make necessary views
        unsup = df[(df.parameters_given) | (df.metric.isin(PARAMLESS_MEASURES))]
        unsup.loc[(unsup.metric == 'dtw-d') & (unsup.metric_params == "{'sakoe_chiba_radius': 0.1}"), 'metric'] = "dtw-d-10"
        unsup.loc[(unsup.metric == 'dtw-d') & (unsup.metric_params == "{'sakoe_chiba_radius': 1.0}"), 'metric'] = "dtw-d-100"
        unsup.loc[(unsup.metric == 'dtw-i') & (unsup.metric_params == "{'sakoe_chiba_radius': 0.1}"), 'metric'] = "dtw-i-10"
        unsup.loc[(unsup.metric == 'dtw-i') & (unsup.metric_params == "{'sakoe_chiba_radius': 1.0}"), 'metric'] = "dtw-i-100"

        norm_experiment(unsup)
        lockstep(df)
        sliding(df)
        elastic(df)
        kernel(df)
        feature(df)
        embedding(df)
        ensemble(df)
        dependency(df)
        runtime(df)