import os
import re
import argparse

def rename_files(directory):
    """
    Rename files in the directory by removing the '+seed42' suffix from .mp4 filenames.
    """
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Only process video files
        if filename.endswith('.mp4'):
            # Use regex to match the +seed42 pattern
            new_name = re.sub(r'\+seed42(?=\.mp4$)', '', filename)

            # If the filename changed, rename it
            if new_name != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_name)
                print(f'Renamed: {filename} -> {new_name}')
                os.rename(old_path, new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove '+seed42' suffix from .mp4 filenames")
    parser.add_argument("--directory", type=str, required=True, help="Path to the video folder")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        exit(1)

    rename_files(args.directory)
    print("Renaming complete!")