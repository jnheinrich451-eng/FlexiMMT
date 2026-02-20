"""
Extract masks from reference videos
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import csv
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import cv2
from utils import find_matching_row
from grounding_sam2 import segment_video, visualize_video_segments

def process_refvideos_in_directory(directory):
    """
    Recursively traverse all videos in the path, compute the CLIP similarity with the matching ori_prompt
    :param directory: Path to the target directory
    :return: List of CLIP similarity for all videos
    """
    # Traverse the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == "crop.csv":
                csv_path = os.path.join(root, file)
                try:
                    with open(csv_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader)  # Get column names
                        path_idx = header.index('path')
                        concept_words_idx = header.index('concept_words')
                        for row in reader:
                            ori_path = row[path_idx]
                            ori_concept_word = row[concept_words_idx]
                            # if ori_path == "benchmark_new/reference_videos/animal/deer_crop/10.mp4":
                            if ori_path.split("/")[-1]=="10.mp4":
                                video_segments = segment_video(
                                    video_path=ori_path,
                                    labels=[ori_concept_word],
                                    threshold=0.5,
                                    polygon_refinement=False,
                                    detector_id="IDEA-Research/grounding-dino-base",
                                    segmenter_id="facebook/sam2.1-hiera-large"
                                )

                                if len(video_segments) != 1:
                                    # Write the erroneous ori_path to a file
                                    error_log_path = "error_extract_ref_paths.txt"
                                    with open(error_log_path, 'a') as error_file:
                                        error_file.write(f"{ori_path}\n")
                                    print(f"Warning: {ori_path}")
                                    continue

                                # Save masks to the specified location
                                for video_segment in video_segments.values():
                                    frames = video_segment["frames"]
                                    base_path = ori_path.replace('reference_videos', 'reference_video_masks_eval')
                                    base_path = '/'.join(base_path.split('/')[0:-2])
                                    label = ori_path.split("/")[-2].removesuffix('_crop')
                                    save_path = os.path.join(base_path, label)
                                    os.makedirs(save_path, exist_ok=True)
                                    for frame_key, frame_value in frames.items():
                                        # Save frame_value as an image to save_path, named as frame_key; frame_value is a uint8 ndarray with values 0 or 255
                                        mask_image = Image.fromarray(frame_value)
                                        mask_path = os.path.join(save_path, f"{frame_key}.png")
                                        mask_image.save(mask_path)

                except Exception as e:
                    print(f"Error processing CSV {csv_path}: {e}")


# Example call
target_directory = "benchmark_new/captions_train/animal" # change to your own output path
all_similarities = process_refvideos_in_directory(target_directory)