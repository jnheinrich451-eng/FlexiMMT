import os
import argparse
import pandas as pd
from pathlib import Path

def process_csv_files(root_folder):
    """
    Iterate over all subfolders under root_folder, read result.csv files,
    compute the mean of the last column, and write to result_all_new.txt.
    """
    root_path = Path(root_folder)

    # Iterate over all subfolders
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue

        csv_file = subfolder / 'result.csv'

        # Check if result.csv exists
        if not csv_file.exists():
            print(f"Warning: result.csv not found in {subfolder.name}")
            continue

        try:
            # Read CSV file
            df = pd.read_csv(csv_file)

            # Get the last column
            last_column = df.iloc[:, -1]

            # Compute the mean
            average_score = last_column.mean()

            # Get the number of rows (video count)
            num_videos = len(df)

            # Write to result_all_new.txt
            output_file = subfolder / 'result_all_new.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Total processed: {num_videos}\n")
                f.write(f"Average score: {average_score:.4f}\n")

            print(f"Processed: {subfolder.name} - Videos: {num_videos}, Average score: {average_score:.4f}")

        except Exception as e:
            print(f"Error processing {subfolder.name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average scores from result.csv files in subfolders")
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder containing subfolders with result.csv files")
    args = parser.parse_args()

    print(f"Processing folder: {args.root_folder}")
    process_csv_files(args.root_folder)
    print("Processing complete!")