import argparse
from pathlib import Path
import pandas as pd

def aggregate_csvs(result_dir: Path, output_dir: Path, output_filename: str = "summary_AD_results.csv"):
    # Only collect CSVs inside subfolders (not directly in result_dir)
    csv_files = [
        csv for csv in result_dir.rglob("*.csv")
        if csv.parent != result_dir and csv.name.endswith(".csv")
    ]

    print(f"Found {len(csv_files)} CSV files for aggregation.")

    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to read {file}: {e}")

    if not all_dfs:
        print("No valid CSVs to aggregate.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    merged_df.to_csv(output_path, index=False)
    print(f"Aggregated CSV written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate CSVs from nested experiment folders.")
    parser.add_argument('-d', '--result_dir', type=str, default='./AD_results', help='Directory containing experiment results')
    parser.add_argument('-o', '--output_dir', type=str, default='./AD_results', help='Directory to write the summary CSV')

    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)

    aggregate_csvs(result_dir=result_dir, output_dir=output_dir)

if __name__ == "__main__":
    main()
