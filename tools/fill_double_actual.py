#!/usr/bin/env python3

import os
import csv
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fill CSV file with image paths from a directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the image files")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--csv_path_prefix", type=str, default=None,
                        help="Prefix for image paths written to CSV (defaults to image_dir)")
    args = parser.parse_args()

    image_dir = args.image_dir
    csv_file = args.csv_file
    csv_path_prefix = args.csv_path_prefix if args.csv_path_prefix else image_dir

    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Error: directory {image_dir} does not exist")
        exit(1)

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        # Also check uppercase extensions
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_files:
        print(f"Warning: no image files found in {image_dir}")
        exit(1)

    # Prepare CSV data
    csv_data = []
    for image_path in sorted(image_files):
        # Extract filename (with extension)
        image_filename = os.path.basename(image_path)
        # Build the path to write into CSV
        csv_entry = f"{csv_path_prefix}/{image_filename}"
        csv_data.append([csv_entry])

    # Read original CSV file, preserving column names
    try:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read the first row (column names)
            original_data = list(reader)  # Read remaining data rows
    except FileNotFoundError:
        print(f"Warning: original CSV file {csv_file} not found, creating a new file")
        header = ["image_path"]  # Default column name, adjust as needed
        original_data = []

    # Write CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write column names

        # Write data rows
        for i, new_row in enumerate(csv_data):
            if i < len(original_data):
                # If original data row exists, replace the first column value
                row = original_data[i].copy()
                row[0] = new_row[0]
                writer.writerow(row)
            else:
                # If original data row does not exist, add a new row
                # Ensure row length matches the number of columns
                while len(new_row) < len(header):
                    new_row.append("")
                writer.writerow(new_row)

    print(f"Success: wrote {len(image_files)} image filenames to {csv_file}, preserving original column names")

if __name__ == "__main__":
    main()
