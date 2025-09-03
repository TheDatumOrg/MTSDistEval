from scripts.anomaly_detection.generate_plots import main as ad_plots
from scripts.classification.generate_plots import main as cls_plots
from scripts.clustering.generate_plots import main as clu_plots
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for different tasks")
    parser.add_argument("--cls_results", help="Path to the classification results")
    parser.add_argument("--clu_results", help="Path to the clustering results")
    parser.add_argument("--ad_results", help="Path to the anomaly detection results")
    parser.add_argument("-o", "--output", help="Path to the output directory", default="./plots")

    args = parser.parse_args()

    OUTDIR = args.output

    # print("Generating the classification plots")
    # cls_plots(input_path=args.cls_results, output_dir=os.path.join(OUTDIR, "classification"))
    # print("\n")

    # print("Generating the clustering plots")
    # clu_plots(input_dir=args.clu_results, output_dir=os.path.join(OUTDIR, "clustering"))
    # print("\n")

    print("Generating the anomaly detection plots")
    ad_plots(input_dir=args.ad_results, output_dir=os.path.join(OUTDIR, "anomaly_detection"))
    print("\n")