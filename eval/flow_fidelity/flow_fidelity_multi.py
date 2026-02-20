import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from raft.core.raft import RAFT
from raft.core.utils.utils import InputPadder
import os
import csv

animal_map = ["bear", "camel", "deer", "cows", "dog", "dogjump", "dogstand", "horsejump-high", "kangaroo"]
human_map = ["chest", "crouch", "fitness", "hike", "human2animal_1", "human2animal_2", "human2animal_3", "one_leg", "rotate", "roll_head", "tennis"]


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)

def load_raft_model(model_path):
    """Load RAFT model"""
    args = argparse.Namespace(small=False, mixed_precision=True, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model = model.module.cuda().eval()
    return model


def compute_flow(model, frame1, frame2):
    """Compute optical flow between two frames"""
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    
    with torch.no_grad():
        flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
    
    return padder.unpad(flow_up)


def get_flow_similarity(flow1, flow2, mask=None):
    """Compute the similarity between two optical flows"""
    # Normalize optical flow vectors
    flow1_norm = flow1 / (torch.norm(flow1, dim=1, keepdim=True) + 1e-6)
    flow2_norm = flow2 / (torch.norm(flow2, dim=1, keepdim=True) + 1e-6)
    
    # Compute cosine similarity
    similarity = torch.sum(flow1_norm * flow2_norm, dim=1)
    
    if mask is not None:
        # Only consider similarity within the mask region
        similarity = similarity * mask
        avg_similarity = similarity.sum() / (mask.sum() + 1e-6)
    else:
        avg_similarity = similarity.mean()
    
    return similarity, avg_similarity

def get_flow_similarity_local_pattern(flow1, flow2, mask1=None, mask2=None):
    """
    Divide the mask region into grids and compare motion statistics for each grid
    """
    # Check if mask is all zeros; if so, return similarity 0 directly
    if mask1 is not None:
        if not (mask1 > 0.5).any():
            print("Warning: mask1 is all zeros, returning similarity 0")
            return None, torch.tensor(0.0)
    
    if mask2 is not None:
        if not (mask2 > 0.5).any():
            print("Warning: mask2 is all zeros, returning similarity 0")
            return None, torch.tensor(0.0)
    
    # Extract optical flow within the mask region
    if mask1 is not None:
        flow1_masked = flow1[:, :, mask1 > 0.5]
    else:
        flow1_masked = flow1.reshape(flow1.shape[0], flow1.shape[1], -1)
    
    if mask2 is not None:
        flow2_masked = flow2[:, :, mask2 > 0.5]
    else:
        flow2_masked = flow2.reshape(flow2.shape[0], flow2.shape[1], -1)
    
    # Compute motion magnitude distribution similarity (using histograms)
    magnitude1 = torch.norm(flow1_masked, dim=1)
    magnitude2 = torch.norm(flow2_masked, dim=1)
    
    # Move tensors to CPU for histogram computation

    magnitude1_cpu = magnitude1.cpu()
    magnitude2_cpu = magnitude2.cpu()

    # # Create histogram (bins=20)
    # max_magnitude = max(magnitude1_cpu.max().item(), magnitude2_cpu.max().item())
    # max_magnitude = max(max_magnitude, 1e-6)  # Prevent zero
    # hist1, _ = torch.histogram(magnitude1_cpu, bins=20, range=(0, max_magnitude))
    # hist2, _ = torch.histogram(magnitude2_cpu, bins=20, range=(0, max_magnitude))

    # # Normalize histograms
    # hist1 = hist1.float() / (hist1.sum() + 1e-6)
    # hist2 = hist2.float() / (hist2.sum() + 1e-6)

    magnitude1_cpu = magnitude1_cpu / (magnitude1_cpu.mean() + 1e-6)
    magnitude2_cpu = magnitude2_cpu / (magnitude2_cpu.mean() + 1e-6)

    # Use 99th percentile to avoid outlier effects
    percentile_99_1 = torch.quantile(magnitude1_cpu, 0.99)
    percentile_99_2 = torch.quantile(magnitude2_cpu, 0.99)
    max_magnitude_norm = max(percentile_99_1.item(), percentile_99_2.item())
    max_magnitude_norm = max(max_magnitude_norm, 1e-6)

    hist1, _ = torch.histogram(magnitude1_cpu, bins=20, range=(0, max_magnitude_norm))
    hist2, _ = torch.histogram(magnitude2_cpu, bins=20, range=(0, max_magnitude_norm))

    # Normalize histograms
    hist1 = hist1.float() / (hist1.sum() + 1e-6)
    hist2 = hist2.float() / (hist2.sum() + 1e-6)

    # Compute histogram correlation
    magnitude_hist_similarity = torch.sum(torch.sqrt(hist1 * hist2))  # Bhattacharyya coefficient

    # Compute direction distribution similarity
    angle1 = torch.atan2(flow1_masked[:, 1], flow1_masked[:, 0])
    angle2 = torch.atan2(flow2_masked[:, 1], flow2_masked[:, 0])

    # Move tensors to CPU for histogram computation
    angle1_cpu = angle1.cpu()
    angle2_cpu = angle2.cpu()

    # Direction histogram
    angle_hist1, _ = torch.histogram(angle1_cpu, bins=18, range=(-np.pi, np.pi))
    angle_hist2, _ = torch.histogram(angle2_cpu, bins=18, range=(-np.pi, np.pi))

    angle_hist1 = angle_hist1.float() / (angle_hist1.sum() + 1e-6)
    angle_hist2 = angle_hist2.float() / (angle_hist2.sum() + 1e-6)

    angle_hist_similarity = torch.sum(torch.sqrt(angle_hist1 * angle_hist2))

    # Combined similarity
    avg_similarity = (magnitude_hist_similarity + angle_hist_similarity) / 2
    
    return None, avg_similarity


def load_mask(mask_path):
    """Load mask"""
    segm_mask = np.array(Image.open(mask_path))
    segm_mask = torch.tensor(segm_mask).float()
    if segm_mask.max() > 1:
        segm_mask = segm_mask / 255
    return segm_mask


def compute_video_flow_similarity(model, video1_path, video2_path, mask1_dir=None, mask2_dir=None, target_size=None):
    """
    Compute the optical flow similarity between two videos

    Args:
        model: RAFT model
        video1_path: Path to the original video
        video2_path: Path to the edited video
        mask1_dir: Mask directory for the original video (containing masks for all frames)
        mask2_dir: Mask directory for the edited video (containing masks for all frames)
        target_size: Target size (height, width)
    """
    # Read videos
    video1 = read_video_from_path(video1_path)
    video2 = read_video_from_path(video2_path)
    
    # Convert to Torch format
    video1 = torch.from_numpy(video1).permute(0, 3, 1, 2).float()  # [T, C, H, W]
    video2 = torch.from_numpy(video2).permute(0, 3, 1, 2).float()  # [T, C, H, W]
    
    # If target size is specified, resize video 1
    if target_size is not None:
        target_height, target_width = target_size
        T, C, H, W = video1.shape
        video1 = torch.nn.functional.interpolate(
            video1,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
    
    video1 = video1.cuda()
    video2 = video2.cuda()
    
    # Load masks for all frames
    masks1 = []
    masks2 = []
    num_frames = min(len(video1), len(video2))
    
    if mask1_dir is not None:
        for i in range(num_frames):
            mask_file = os.path.join(mask1_dir, f"{i}.png")
            if os.path.exists(mask_file):
                mask = load_mask(mask_file)
                # Resize to target size
                if target_size is not None:
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=target_size,
                        mode='nearest'
                    ).squeeze()
                masks1.append(mask)
            else:
                masks1.append(None)
    
    if mask2_dir is not None:
        for i in range(num_frames):
            mask_file = os.path.join(mask2_dir, f"{i}.png")
            if os.path.exists(mask_file):
                mask = load_mask(mask_file)
                # Resize to target size
                if target_size is not None:
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=target_size,
                        mode='nearest'
                    ).squeeze()
                masks2.append(mask)
            else:
                masks2.append(None)
    
    # Compute optical flow between each pair of consecutive frames
    flow_similarities = []
    for i in range(num_frames - 1):
        flow1 = compute_flow(model, video1[i:i+1], video1[i+1:i+2])
        flow2 = compute_flow(model, video2[i:i+1], video2[i+1:i+2])
        
        # Use the current frame's mask
        mask1_i = masks1[i] if mask1_dir is not None and i < len(masks1) else None
        mask2_i = masks2[i] if mask2_dir is not None and i < len(masks2) else None
        
        _, frame_similarity = get_flow_similarity_local_pattern(flow1, flow2, mask1_i, mask2_i)
        flow_similarities.append(frame_similarity.item())
    
    return {
        "frame_similarities": flow_similarities,
        "average_similarity": np.mean(flow_similarities)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, 
                       default="eval/flow_fidelity/configs/flow_fidelity_score_config_multi.yaml")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)
    
    # Load RAFT model
    raft_model = load_raft_model(config.raft_model_path)
    
    # Create output directory
    output_summary_dir = Path(config.output_path)
    output_summary_dir.mkdir(parents=True, exist_ok=True)
    
    result_all_path = output_summary_dir / "result_all.txt"
    result_csv_path = output_summary_dir / "result.csv"
    
    # Store results for all videos
    all_video_results = []
    all_average_scores = []
    
    # Set paths
    csv_directory = config.csv_directory
    video_directory = os.path.join(config.edit_root_path, "videos")
    original_videos_dir = config.original_videos_path
    original_masks_dir = config.original_masks_path
    
    # Iterate over CSV files
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
            concept_words_idx = header.index('concept_words')
            
            # Read each row of data
            for row in reader:
                if len(row) <= max(path_idx, sources_idx, caption_idx, concept_words_idx):
                    continue
                    
                img_name = row[path_idx].split("/")[-1]
                sources_name = row[sources_idx]
                ori_caption = row[caption_idx]
                ori_concept_words = row[concept_words_idx]
                
                # Build video filename and path
                video_filename = f"{img_name}+{sources_name}.mp4"
                video_path = os.path.join(video_directory, video_filename)
                file_name_without_extension = f"{img_name}+{sources_name}"
                
                # Check if video file exists
                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue
                
                print(f"Processing: {video_path}")

                # Read original video and original video masks
                original_video_paths = []
                original_mask_paths = []
                for source in sources_name.split("+"):
                    if source in animal_map:
                        concat = "animal"
                    elif source in human_map:
                        concat = "human"
                    else:
                        print(f"Unknown source: {source}, skipping")
                        continue
                    original_mask_path = os.path.join(original_masks_dir, concat, source)
                    original_video_path = os.path.join(original_videos_dir, concat, source+"_crop", "10.mp4")
                    original_mask_paths.append(original_mask_path)
                    original_video_paths.append(original_video_path)
                
                # Read target video and target video masks
                edit_video_path = video_path
                edit_mask_paths = []
                for ori_concept_word in ori_concept_words.split("+"):
                    edit_mask_path = os.path.join(config.edit_root_path, file_name_without_extension, ori_concept_word)
                    edit_mask_paths.append(edit_mask_path)
                
                # Get the dimensions of the edited video
                try:
                    edit_video = read_video_from_path(edit_video_path)
                    edit_height, edit_width = edit_video.shape[1:3]
                    print(f"  Edit video size: {edit_width}x{edit_height}")
                except Exception as e:
                    print(f"  Error reading video: {str(e)}")
                    continue

                # Compute similarity
                all_scores = []
                all_frame_similarities = []
                
                # Iterate over each source video and corresponding mask
                for idx, (original_video_path, original_mask_path, edit_mask_path) in enumerate(
                    zip(original_video_paths, original_mask_paths, edit_mask_paths)
                ):
                    print(f"  Processing object {idx+1}/{len(original_video_paths)}")

                    # Compute optical flow similarity
                    try:
                        similarity_scores_dict = compute_video_flow_similarity(
                            raft_model,
                            original_video_path,
                            edit_video_path,
                            mask1_dir=original_mask_path if config.use_mask else None,
                            mask2_dir=edit_mask_path if config.use_mask else None,
                            target_size=(edit_height, edit_width)
                        )
                        
                        all_scores.append(similarity_scores_dict["average_similarity"])
                        all_frame_similarities.append(similarity_scores_dict["frame_similarities"])
                        
                        print(f"    Object {idx+1} score: {similarity_scores_dict['average_similarity']:.4f}")
                    except Exception as e:
                        print(f"  Error processing object {idx+1}: {str(e)}")
                        continue
                
                # Save results for the current video
                if all_scores:
                    average_score = np.mean(all_scores)
                    
                    # Record results for summary
                    all_video_results.append({
                        'video_name': file_name_without_extension,
                        'video_path': video_path,
                        'num_objects': len(all_scores),
                        'individual_scores': all_scores,
                        'flow_fidelity': average_score
                    })
                    all_average_scores.append(average_score)
                    
                    print(f"  Average score: {average_score:.4f}")
                else:
                    print(f"  No valid scores computed for this video\n")
    
    # Write overall results to result_all.txt
    with open(result_all_path, 'w', encoding='utf-8') as f:
        f.write(f"Total videos processed: {len(all_average_scores)}\n")
        if all_average_scores:
            overall_avg_score = np.mean(all_average_scores)
            f.write(f"\nOverall average flow fidelity score: {overall_avg_score:.4f}\n")
            print(f"\nOverall average flow fidelity score: {overall_avg_score:.4f}")
        else:
            f.write("\nNo videos processed.\n")
            print("\nNo videos processed.")
    
    # Write individual video results to result.csv
    if all_video_results:
        with open(result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_name', 'video_path', 'num_objects', 'individual_scores', 'flow_fidelity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_video_results:
                # Convert individual_scores to string for CSV writing
                result_copy = result.copy()
                result_copy['individual_scores'] = str(result['individual_scores'])
                writer.writerow(result_copy)
    
    print(f"\nResults saved to:")
    print(f"  Overall results: {result_all_path}")
    print(f"  Detailed results: {result_csv_path}")