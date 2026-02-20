import os
import csv
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

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

def calculate_pairwise_cosine_similarity(features):
    """
    Compute the pairwise cosine similarity of all frames
    :param features: CLIP feature matrix (n_frames, feature_dim)
    :return: Average pairwise cosine similarity
    """
    if len(features) < 2:
        return 0.0  # If there are less than 2 frames, return 0
    
    # Compute the cosine similarity matrix of all frames
    similarity_matrix = cosine_similarity(features)
    
    # Extract the upper triangle (excluding the diagonal)
    upper_triangle = np.triu(similarity_matrix, k=1)
    
    # Compute the average of non-zero elements
    avg_similarity = np.sum(upper_triangle) / np.count_nonzero(upper_triangle)
    
    return avg_similarity

def process_video(video_path):
    """
    Process a single video, compute the pairwise cosine similarity of all frames
    :param video_path: Path to the video
    :return: Average pairwise cosine similarity
    """
    # Extract video frames
    frames = extract_frames(video_path)
    if not frames:
        print(f"No frames extracted from {video_path}")
        return 0.0
    
    # Compute CLIP features
    features = compute_clip_features(frames)
    
    # Compute the pairwise cosine similarity of all frames
    avg_pairwise_similarity = calculate_pairwise_cosine_similarity(features)
    return avg_pairwise_similarity

def process_videos_in_csv_directory(csv_directory, video_directory, output_dir="eval_outputs/temporal_consistency"):
    """
    Process videos based on CSV files, compute temporal consistency for each video
    :param csv_directory: Directory containing CSV files
    :param video_directory: Directory containing video files
    :param output_dir: Output file directory
    :return: List of temporal consistency scores for all videos
    """
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

            # Read each row of data
            for row in reader:
                if len(row) <= max(path_idx, sources_idx):
                    continue

                img_name = row[path_idx].split("/")[-1]
                sources_name = row[sources_idx]

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
                    avg_pairwise_similarity = process_video(video_path)
                    all_similarities.append(avg_pairwise_similarity)

                    video_results.append({
                        'video_path': video_path,
                        'video_name': video_filename,
                        'temporal_consistency': avg_pairwise_similarity
                    })

                    print(f"Average pairwise cosine similarity: {avg_pairwise_similarity:.4f}")
                except Exception as e:
                    print(f"Failed to process {video_path}: {e}")

    # Write overall results to result_all.txt
    with open(result_all_path, 'w', encoding='utf-8') as f:
        f.write(f"Total videos processed: {len(all_similarities)}\n")
        # f.write(f"All similarities: {all_similarities}\n")
        if all_similarities:
            overall_avg_similarity = np.mean(all_similarities)
            f.write(f"\nOverall average pairwise cosine similarity: {overall_avg_similarity:.4f}\n")
            print(f"Overall average pairwise cosine similarity: {overall_avg_similarity:.4f}")
        else:
            f.write("\nNo videos processed.\n")
            print("No videos processed.")

    # Write individual video results to result.csv
    if video_results:
        with open(result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_path', 'video_name', 'temporal_consistency']
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
output_dir = "eval_output_mask_dynamic_timestep_token_nlastframe1/temporal_consistency"
all_similarities = process_videos_in_csv_directory(target_csv_directory, target_video_directory, output_dir)