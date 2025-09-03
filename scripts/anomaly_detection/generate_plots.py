import sys
import argparse
from scripts.anomaly_detection.aggregate_results import aggregate_results
from scripts.anomaly_detection.result_analysis import result_analysis

def main(input_dir, output_dir):
    # Main function to generate plots
    concat_path = "/tmp/AD_results.csv"

    # Run aggregate results through command line
    aggregate_results(result_dir=input_dir, output_path=concat_path)
    result_analysis(input_path=concat_path, output_path=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for anomaly detection results")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input directory for results")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path for results")

    args = parser.parse_args()

    results_dir = args.input
    output_dir = args.output

    main(results_dir, output_dir)