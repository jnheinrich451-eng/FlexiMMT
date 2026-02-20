import os
import csv

def find_matching_row(csv_dir, img_name, source_name):
    """
    Find a matching row in the CSV directory
    :param csv_dir: Path to the CSV directory
    :param img_name: Image file name
    :param source_name: Source name
    :return: Matching row data, or None if not found
    """
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_path = os.path.join(root, file)
                try:
                    with open(csv_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader)  # Get column names

                        # Ensure the CSV format is correct
                        if 'path' in header and 'sources' in header and 'caption' in header:
                            path_idx = header.index('path')
                            sources_idx = header.index('sources')
                            caption_idx = header.index('caption')

                            for row in reader:
                                if len(row) > max(path_idx, sources_idx, caption_idx):
                                    # Extract file name from path
                                    path = row[path_idx]
                                    file_name = path.split('/')[-1]

                                    # Check if file name and sources match
                                    if file_name == img_name and row[sources_idx] == source_name:
                                        return row, header
                except Exception as e:
                    print(f"Error processing CSV {csv_path}: {e}")

    return None, header