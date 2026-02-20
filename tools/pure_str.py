import os
import re
import argparse

def rename_files(directory):
    """
    Rename files in the directory from 'xxx_yyy' format to 'yyy',
    where xxx is a numeric prefix and yyy may contain underscores.
    """
    pattern = r'^[0-9]+_(.+)$'

    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    renamed_count = 0
    skipped_count = 0

    for filename in files:
        match = re.match(pattern, filename)
        if match:
            # Extract the yyy part
            new_name = match.group(1)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            # Check if the target filename already exists
            if os.path.exists(new_path):
                print(f"Skipped: {filename} -> {new_name} (target file already exists)")
                skipped_count += 1
            else:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
                renamed_count += 1
        else:
            print(f"Skipped: {filename} (does not match pattern)")
            skipped_count += 1

    print(f"\nDone! Renamed {renamed_count} files, skipped {skipped_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove numeric prefix from filenames (e.g. '001_name.png' -> 'name.png')")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing files to rename")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        exit(1)

    rename_files(args.directory)