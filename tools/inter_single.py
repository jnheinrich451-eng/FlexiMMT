import os
import pandas as pd
from pathlib import Path

def process_folders(base_dir, output_csv_path):
    """
    Process the animal and human folders under base_dir, merging val_image.csv files from all subfolders.
    """
    main_folders = ["animal", "human"]
    all_data = []
    
    for main_folder in main_folders:
        main_path = os.path.join(base_dir, main_folder)
        if not os.path.exists(main_path):
            print(f"Folder does not exist: {main_path}")
            continue

        # Get all subfolders
        subfolders = [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]

        for subfolder in subfolders:
            subfolder_path = os.path.join(main_path, subfolder)
            csv_path = os.path.join(subfolder_path, "val_image.csv")

            if os.path.exists(csv_path):
                try:
                    # Read CSV file
                    df = pd.read_csv(csv_path)

                    # Add sources column
                    df['sources'] = subfolder

                    # Reorder columns
                    if all(['path' in df.columns, 'caption' in df.columns,
                           'concept_words' in df.columns, 'motion_words' in df.columns]):
                        df = df[['path', 'sources', 'caption', 'concept_words', 'motion_words']]
                        all_data.append(df)
                    else:
                        print(f"CSV file {csv_path} has unexpected column names")
                except Exception as e:
                    print(f"Error processing file {csv_path}: {str(e)}")
            else:
                print(f"CSV file does not exist: {csv_path}")

    if all_data:
        # Merge all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save to output file
        combined_df.to_csv(output_csv_path, index=False)
        print(f"Merge complete, saved to: {output_csv_path}")
    else:
        print("No valid CSV files found")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge val_image.csv files from animal/human subfolders")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing animal/human folders (e.g. benchmark_new/captions_full)")
    parser.add_argument("--output_csv", type=str, required=True, help="Output path for the merged CSV file")
    args = parser.parse_args()
    process_folders(args.base_dir, args.output_csv)