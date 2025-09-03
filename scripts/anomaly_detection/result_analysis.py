import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from scikit_posthocs import posthoc_nemenyi_friedman
from scripts.Friedman_Nemenyi_test import graph_ranks
import sys


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
        print("P-value for wilcoxon test: ", pval)
    else:
        # Compute friedman p-value
        pval = friedmanchisquare(*pv.values)[1]
        print("P-value for friedman test: ", pval)

    # If significant, do nemenyi test
    if pval < alpha:
        print("Significant difference between treatments, doing Nemenyi test")
        nemenyi = posthoc_nemenyi_friedman(pv.T.values).to_numpy()
    else:
        # Create dummy nemenyi with high p-values
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
            return pd.Series([0,0,0], index=['greater', 'less', 'diff_p'])
    
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

def all_to_all_matrix(df, ref_metric, metric='metric', norm='norm', problem='problem', acc='acc', alpha=0.1, plot=True, figsize=(12, 8), transpose=False):
    # Create left dataset
    left = df[df[metric] == ref_metric][[metric, norm, problem, acc]].set_index([problem, norm])
    right = df[df[metric] != ref_metric][[metric, norm, problem, acc]].set_index([problem, norm])

    # Join left with right on problem and normalization
    join = left.join(right, lsuffix='_left', rsuffix='_right', how='inner').reset_index()

    matrix = join.groupby([norm, metric + '_right']).apply(wilcox_p)

    ax = None
    if plot:
        wmatrix = matrix.unstack()['p_value']

        if transpose:
            wmatrix = wmatrix.T

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(10, 133, as_cmap=True)

        # Increase fontsize
        plt.rcParams.update({'font.size': 16})

        # Create heatmap of matrix
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(wmatrix, annot=True, cmap=cmap, lw=1, vmin=-1, vmax=1, cbar=False, ax=ax)

        if transpose:
            ax.set_xlabel(ref_metric)
            ax.set_ylabel("Comparison metric")
        else:
            ax.set_ylabel(ref_metric)
            ax.set_xlabel("Comparison metric")
      

        # Create a custom legend with 1 for better, -1 for worse, 0 for equal
        handles = [plt.Rectangle((0,0),1,1, color=cmap.get_over(), label='Better'),
                   plt.Rectangle((0,0),1,1, color='white', label='Equal'),
                    plt.Rectangle((0,0),1,1, color=cmap.get_under(), label='Worse')]
        legend = ax.legend(handles, ['Better', 'Equal', 'Worse'], title='Significance', loc='upper right', bbox_to_anchor=(1.3, 1))
        legend.get_frame().set_facecolor('lightgrey')
        
    return matrix, ax

def norm_comparison(df, refs, treatment='norm', category='metric', problem='problem', acc='acc', alpha=0.1, plot=True, figsize=(12, 8), transpose=False):
    # Create left dataset
    left = df[df[treatment].isin(refs)][[treatment, category, problem, acc]].set_index([problem, category])
    right = df[[treatment, category, problem, acc]].set_index([problem, category])

    # Join left with right on problem and normalization
    join = left.join(right, lsuffix='_left', rsuffix='_right', how='inner').reset_index()

    join = join[join[f"{treatment}_left"] != join[f"{treatment}_right"]]

    # Perform wilcoxon for all pairs
    matrix = join.groupby([treatment + '_left', treatment + '_right']).apply(lambda g: wilcox_p(g, acc=acc, alpha=alpha))

    ax = None
    if plot:
        wmatrix = matrix.unstack()['p_value']

        if transpose:
            wmatrix = wmatrix.T


        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(10, 133, as_cmap=True)

        # Increase fontsize
        plt.rcParams.update({'font.size': 16})

        # Create heatmap of matrix
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(wmatrix, annot=True, cmap=cmap, lw=1, cbar=False, ax=ax)

        ax.set_xlabel("")
        ax.set_ylabel("")

        # Create a custom legend with 1 for better, -1 for worse, 0 for equal
        handles = [plt.Rectangle((0,0),1,1, color=cmap.get_over(), label='Better'),
                   plt.Rectangle((0,0),1,1, color='white', label='Equal'),
                    plt.Rectangle((0,0),1,1, color=cmap.get_under(), label='Worse')]
        legend = ax.legend(handles, ['Better', 'Equal', 'Worse'], title='Significance', loc='upper right', bbox_to_anchor=(1.3, 1))
        legend.get_frame().set_facecolor('lightgrey')
        
    return matrix, ax

def mat_to_latex(matrix):
    # Set first index as column
    tmp = matrix.reset_index().pivot(index='norm_formal_right', columns='norm_formal_left')

    # Switch first and second level columns
    tmp.columns = tmp.columns.swaplevel(0, 1)

    tmp = tmp[['Nonorm', 'Z-score']]

    tmp = tmp.fillna(-1000).astype(int)

    ltx = tmp.to_latex()

    ltx = ltx.replace("norm_formal_left", "") \
            .replace("norm_formal_right", "") \
            .replace("-1000", "") \
            .replace(" -1 ", " \\xmark ") \
            .replace(" 0 ", " \\omark ") \
            .replace(" 1 ", " \\cmark ") \
            .replace("toprule", "hline") \
            .replace("midrule", "hline") \
            .replace("bottomrule", "hline") \
            .replace("<", "$\\mathbf{<}$") \
            .replace(">", "$\\mathbf{>}$") \
            .replace("=", "$\\mathbf{=}$") \
            .replace("p_value", "\\textbf{Diff.}") \
            .replace("{lrrrrrrrr}", "{|l|rrrr|rrrr|}") \
            .replace("\\multicolumn{4}{r}", "\\multicolumn{4}{c|}") \
            .replace("&  &  &  &  &  &  &  &  \\\\", "")

    return ltx


def cdwrapper(g, alpha=0.1):
    metric = g.name
    print("Metric: ", metric)
    stat_test(g, treatment="normalization", datasets="dataset", acc="accuracy", alpha=alpha)

    # Do the type analysis with znorm as reference
    ref = g[g.normalization == 'zscore'].set_index('dataset').accuracy.to_dict()

    # Remove the reference from the df
    g = g[g.normalization != 'zscore']

    print(type_analysis(g, ref, ref_name=(metric, 'zscore'), alpha=alpha))
    plt.savefig(f'figures/{metric}_cd.pdf')

def result_analysis(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # set up params
    metric_name = 'VUS-PR'
    ALPHA=0.1
    NORM_ORDER = ["nonorm"]

    # load data
    data_df = pd.read_csv(input_path)
    if 'classifier_name' in data_df.columns:
        data_df = data_df.rename(columns={'classifier_name': 'metric', 
                                          'dataset_name': 'problem', 
                                          'accuracy': 'acc'})

    else:
        data_df = data_df.rename(columns={'distance_measure': 'metric', 
                                        metric_name: 'acc'})
        
    data_df["metric"] = data_df["metric"].replace("euclidean", "Euclidean")

    data_df['norm'] = 'nonorm'

    ed_df = data_df[data_df.metric == 'Euclidean'].set_index("problem").acc.to_dict()

    # analysis
    AD_stats = type_analysis(data_df[data_df.metric!= 'Euclidean'], ed_df, ref_name=('Euclidean', 'nonorm'), alpha=ALPHA)

    original_stdout = sys.stdout
    with open(os.path.join(output_path, 'tables.txt'), 'w') as f:
        sys.stdout = f
        print("Table 10: Pairwise comparison of lock-step, sliding, and elastic measures on the TSB-AD-M archive, using a 1NN anomaly detector. See Table 3 for column descriptions.")
        print(AD_stats)
    sys.stdout = original_stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input CSV file (aggregated csv)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path for results")

    args = parser.parse_args()

    result_analysis(args.input, args.output)
