import os
import csv
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import cv2
from utils import find_matching_row

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frames(video_path, max_frames=None):
    """
    Extract frames from a video
    :param video_path: Path to the video
    :param max_frames: Maximum number of frames to extract (None for all frames)
    :return: List of frames (PIL images)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_count >= max_frames:
            break

        # Convert OpenCV BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL image
        frame_pil = Image.fromarray(frame_rgb)
        frames.append(frame_pil)
        frame_count += 1

    cap.release()
    return frames

def compute_clip_features(images):
    """
    Compute CLIP image features
    :param images: List of PIL images
    :return: CLIP feature matrix (n_frames, feature_dim)
    """
    # Preprocess images and stack into a tensor
    images_preprocessed = torch.stack([preprocess(img) for img in images]).to(device)
    
    # Compute CLIP features
    with torch.no_grad():
        features = model.encode_image(images_preprocessed)
    
    return features.cpu().numpy()

def compute_text_clip_features(text):
    """
    Compute CLIP text features
    :param text: Text
    :return: CLIP text feature (1, feature_dim)
    """
    # Clip text length (CLIP's maximum length is 77)
    text = text[:77]
    
    # Compute CLIP text features
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_tokens)
    
    return text_features.cpu().numpy()

def calculate_text_clip_similarity(video_path, ori_prompt):
    """
    Compute the CLIP similarity between video frames and text
    :param video_path: Path to the video
    :param ori_prompt: Original text prompt
    :return: Average CLIP similarity
    """
    # Extract video frames
    frames = extract_frames(video_path)
    if not frames:
        print(f"No frames extracted from {video_path}")
        return 0.0
    
    # Compute the CLIP features of the video frames
    frame_features = compute_clip_features(frames)
    
    # Compute the CLIP features of the text
    text_features = compute_text_clip_features(ori_prompt)
    
    # Compute the cosine similarity between the video frames and the text
    similarities = cosine_similarity(text_features, frame_features)
    
    # Return the average similarity
    return np.mean(similarities)

def process_videos_in_csv_directory(csv_directory, video_directory, output_dir="eval_outputs/text_similarity"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Output file paths
    result_all_path = os.path.join(output_dir, "result_all.txt")
    result_csv_path = os.path.join(output_dir, "result.csv")

    # Read each row from csv
    all_similarities = []
    video_results = []

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
            caption_idx = header.index('caption')

            # Read each row of data
            for row in reader:
                if len(row) <= max(path_idx, sources_idx, caption_idx):
                    continue

                img_name = row[path_idx].split("/")[-1]
                sources_name = row[sources_idx]
                caption = row[caption_idx]

                # Read the corresponding video
                # Construct video filename: img_name+source_name.mp4
                video_filename = f"{img_name}+{sources_name}.mp4"
                video_path = os.path.join(video_directory, "videos", video_filename)

                # Check if video file exists
                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue

                print(f"Processing: {video_path}")

                try:
                    # Compute and save results
                    avg_similarity = calculate_text_clip_similarity(video_path, caption)
                    all_similarities.append(avg_similarity)

                    video_results.append({
                        'video_path': video_path,
                        'caption': caption,
                        'clip_similarity': avg_similarity
                    })

                    print(f"Average CLIP similarity: {avg_similarity:.4f}")
                except Exception as e:
                    print(f"Failed to process {video_path}: {e}")

    # Write overall results to result_all.txt
    with open(result_all_path, 'w', encoding='utf-8') as f:
        f.write(f"Total videos processed: {len(all_similarities)}\n")
        # f.write(f"All similarities: {all_similarities}\n")
        if all_similarities:
            overall_avg_similarity = np.mean(all_similarities)
            f.write(f"\nOverall average CLIP similarity: {overall_avg_similarity:.4f}\n")
            print(f"Overall average CLIP similarity: {overall_avg_similarity:.4f}")
        else:
            f.write("\nNo videos processed.\n")
            print("No videos processed.")

    # Write individual video results to result.csv
    if video_results:
        with open(result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_path', 'caption', 'clip_similarity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in video_results:
                writer.writerow(result)

    print(f"\nResults saved to:")
    print(f"  Overall results: {result_all_path}")
    print(f"  Detailed results: {result_csv_path}")

    return all_similarities

# Example call
target_csv_directory = "benchmark_new/captions_inf_all"
target_video_directory = "outputs_mask_dynamic_timestep_token_nlastframe1" # change to your own output path
output_dir = "eval_output_mask_dynamic_timestep_token_nlastframe1/text_similarity"
all_similarities = process_videos_in_csv_directory(target_csv_directory, target_video_directory, output_dir)