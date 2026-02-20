import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import csv
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import cv2
from scripts.utils import get_gt_masks

def compute_masks_in_csv(csv_directory):
    for root, dirs, files in os.walk(csv_directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_path = os.path.join(root, file)
                try:
                    with open(csv_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        header = next(reader)  # Get column names

                        for row in reader:
                            ori_concept_words = row[header.index('concept_words')].split("+")
                            ori_path = row[header.index('path')]
                            ori_sources = row[header.index('sources')]
                            ori_img = Image.open(ori_path).convert("RGB")

                            # Replace 'target_images' with 'target_masks' in the path
                            concept_mask_base_path = ori_path.replace('target_images', 'target_masks')
                            concept_mask_base_path = concept_mask_base_path + "+" + ori_sources
                            # concept_mask_base_path = os.path.splitext(concept_mask_base_path)[0]  # Remove extension

                            # Skip if concept_mask_base_path already exists and is complete
                            if os.path.exists(concept_mask_base_path) and len(os.listdir(concept_mask_base_path))==len(ori_concept_words):
                                print(f"Skipping {ori_path}: masks already exist")
                                continue

                            # Extract mask for each object based on concept_words
                            gt_masks = get_gt_masks(ori_img, ori_img.height, ori_img.width, ori_concept_words)

                            if gt_masks.shape[0] != len(ori_concept_words):
                                # Log the erroneous ori_path to a file
                                error_log_path = os.path.join(csv_directory, "error_extract_mask_paths.txt")
                                with open(error_log_path, 'a') as error_file:
                                    error_file.write(f"{ori_path}\n")
                                print(f"Warning: mask count ({gt_masks.shape[0]}) != concept_words count ({len(ori_concept_words)}) for {ori_path}")
                                continue
                            
                            # Create directory for saving masks
                            os.makedirs(concept_mask_base_path, exist_ok=True)

                            # Save extracted masks to the specified location
                            for idx, ori_concept_word in enumerate(ori_concept_words):
                                # Get the corresponding mask (bool tensor)
                                mask = gt_masks[idx]  # shape: [height, width]
                                
                                # Convert bool tensor to 0-255 numpy array
                                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                                
                                # Convert to PIL Image and save
                                mask_img = Image.fromarray(mask_np)
                                mask_save_path = os.path.join(concept_mask_base_path, f"{ori_concept_word}.png")
                                mask_img.save(mask_save_path)
                                
                            print(f"Saved masks to {concept_mask_base_path}")

                except Exception as e:
                    print(f"Error processing CSV {concept_mask_base_path}: {e}")

# Example call
csv_directory = "benchmark_new/captions_inf_rebuttal_overlap" # change to your own output path


compute_masks_in_csv(csv_directory)