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

def process_videos_in_csv_directory(csv_directory, video_directory, output_dir="outputs_test_pre"):
    """
    Process videos based on CSV files, extract and save masks
    :param csv_directory: Directory containing CSV files
    :param video_directory: Directory containing video files
    :param output_dir: Output file directory (for saving masks)
    :return: Number of processed videos
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    failed_videos = []  # Track failed videos

    # Iterate over all CSV files in csv_directory
    for csv_file in os.listdir(csv_directory):
        if not csv_file.endswith('.csv'):
            continue

        csv_path = os.path.join(csv_directory, csv_file)
        print(f"Processing CSV: {csv_path}")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read header

            # Get required column indices
            path_idx = header.index('path')
            sources_idx = header.index('sources')
            concept_words_idx = header.index('concept_words')

            # Read each row of data
            for row in reader:
                if len(row) <= max(path_idx, sources_idx, concept_words_idx):
                    continue

                img_name = row[path_idx].split("/")[-1]
                sources_name = row[sources_idx]
                ori_concept_words = row[concept_words_idx]
                ori_path = row[path_idx]

                # Construct video filename: img_name+sources_name.mp4
                video_filename = f"{img_name}+{sources_name}.mp4"
                video_path = os.path.join(video_directory, "videos", video_filename)

                # Check if video file exists
                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    failed_videos.append(f"{video_filename}\tReason: file not found")
                    continue

                print(f"Processing: {video_path}")

                # Get filename without extension
                file_name_without_extension = video_filename.rsplit('.', 1)[0]

                should_skip = True
                for concept_words in ori_concept_words.split("+"):
                    pre_path = os.path.join(output_dir, file_name_without_extension, concept_words)
                    if not os.path.exists(pre_path) or len(os.listdir(pre_path)) != 49:
                        should_skip = False
                        break

                if should_skip:
                    print(f"Skipping already processed video (all masks exist and complete): {video_path}")
                    continue

                # Try to load pre-extracted masks
                concept_mask_base_path = ori_path.replace('target_images', 'target_masks')
                concept_mask_base_path = concept_mask_base_path + "+" + sources_name

                first_frame_masks = None  # Initialize
                if os.path.exists(concept_mask_base_path):
                    # concept_mask_base_path contains black-and-white mask images named {concept_word}.png
                    # After loading, they become bool type torch.Size([num_masks, height, width]), where mask order matches concept_words
                    gt_masks = []
                    concept_words_list = ori_concept_words.split("+")

                    for concept_word in concept_words_list:
                        mask_path = os.path.join(concept_mask_base_path, f"{concept_word}.png")
                        if os.path.exists(mask_path):
                            mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
                            # Do not resize, keep original first frame dimensions
                            mask_array = np.array(mask_img)  # Keep original uint8 format
                            gt_masks.append(mask_array)
                        else:
                            print(f"Warning: mask file not found {mask_path}")
                            gt_masks = []  # If any mask is missing, clear list and fall back to auto detection
                            break

                    if gt_masks:
                        # Build first_frame_masks dict, keys are object IDs (starting from 1), values are mask arrays
                        first_frame_masks = {i+1: mask for i, mask in enumerate(gt_masks)}
                        print(f"Successfully loaded {len(gt_masks)} predefined masks for the first frame")



                try:
                    # Call segment_video based on whether predefined masks are available
                    if first_frame_masks is not None:
                        # Use predefined first frame masks
                        video_segments = segment_video(
                            video_path=video_path,
                            first_frame_masks=first_frame_masks,
                            polygon_refinement=False,
                            segmenter_id="facebook/sam2.1-hiera-large"
                        )

                        # Update label info (default labels are object_X when using mask input)
                        concept_words_list = ori_concept_words.split("+")
                        for obj_idx, video_segment in video_segments.items():
                            if obj_idx <= len(concept_words_list):
                                video_segment["label"] = concept_words_list[obj_idx-1]
                    else:
                        # Use automatic detection
                        video_segments = segment_video(
                            video_path=video_path,
                            labels=ori_concept_words.split("+"),
                            threshold=0.4,
                            polygon_refinement=False,
                            detector_id="IDEA-Research/grounding-dino-base",
                            segmenter_id="facebook/sam2.1-hiera-large"
                        )

                    # Save masks to the specified location
                    for video_segment in video_segments.values():
                        label = video_segment["label"]
                        frames = video_segment["frames"]
                        save_path = os.path.join(output_dir, file_name_without_extension, label.rstrip("."))
                        os.makedirs(save_path, exist_ok=True)
                        for frame_key, frame_value in frames.items():
                            # Save frame_value as an image to save_path, named as frame_key; frame_value is a uint8 ndarray with values 0 or 255
                            mask_image = Image.fromarray(frame_value)
                            mask_path = os.path.join(save_path, f"{frame_key}.png")
                            mask_image.save(mask_path)

                    print(f"Finish mask extraction: {video_path}")
                    processed_count += 1

                except Exception as e:
                    error_msg = f"{video_filename}\tReason: {str(e)}"
                    print(f"Failed to process {video_path}: {e}")
                    failed_videos.append(error_msg)

    # Write failed videos to file
    failed_log_path = "failed_extract_masks_output.txt"
    with open(failed_log_path, 'w', encoding='utf-8') as f:
        f.write(f"List of failed videos ({len(failed_videos)} total)\n")
        f.write("=" * 80 + "\n")
        for failed_video in failed_videos:
            f.write(failed_video + "\n")

    print(f"\nTotal processed: {processed_count} videos")
    print(f"Failed: {len(failed_videos)} videos, details saved to: {failed_log_path}")
    return processed_count

# Example call
target_csv_directory = "benchmark_new/captions_inf_all"
target_video_directory = "outputs_mask_dynamic_timestep_token_nlastframe2_k15_gamma0.5"  # change to your own output path
processed_count = process_videos_in_csv_directory(target_csv_directory, target_video_directory, target_video_directory)