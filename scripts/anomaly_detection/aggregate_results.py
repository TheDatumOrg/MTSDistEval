import argparse
import os
from pathlib import Path
import pandas as pd

def aggregate_results(result_dir: Path, output_path: Path):
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

    os.makedirs(output_path, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Aggregated CSV written to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate CSVs from nested experiment folders.")
    parser.add_argument('-d', '--result_dir', type=str, help='Directory containing experiment results')
    parser.add_argument('-o', '--output_path', type=str, help='Path to write the concatenated CSV')

    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    output_path = Path(args.output_path)

    aggregate_results(result_dir=result_dir, output_path=output_path)
