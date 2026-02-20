import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from co_tracker.cotracker.predictor import CoTrackerPredictor
from co_tracker.cotracker.utils.visualizer import read_video_from_path
from einops import rearrange
from omegaconf import OmegaConf
from eval.utils import find_matching_row
import os
import csv

animal_map = ["bear", "camel", "deer", "cows", "dog", "dogjump", "dogstand", "horsejump-high", "kangaroo"]
human_map = ["chest", "crouch", "fitness", "hike", "human2animal_1", "human2animal_2", "human2animal_3", "one_leg", "rotate", "roll_head", "tennis"]

def get_similarity_matrix(tracklets1, tracklets2):
    # Align frame counts
    tracklets1, tracklets2 = align_tracklets_frames(tracklets1, tracklets2)
    
    displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
    displacements1 = displacements1 / (displacements1.norm(dim=-1, keepdim=True) + 1e-8)

    displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
    displacements2 = displacements2 / (displacements2.norm(dim=-1, keepdim=True) + 1e-8)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", displacements1, displacements2).mean(dim=-1)
    return similarity_matrix

def align_tracklets_frames(tracklets1, tracklets2):
    """Truncate two tracklets to the smaller frame count"""
    n1, t1, c = tracklets1.shape
    n2, t2, c = tracklets2.shape
    
    min_frames = min(t1, t2)
    
    # Truncate to the minimum frame count
    tracklets1_aligned = tracklets1[:, :min_frames, :]
    tracklets2_aligned = tracklets2[:, :min_frames, :]
    
    return tracklets1_aligned, tracklets2_aligned


def get_score(similarity_matrix):
    max_similarity, _ = similarity_matrix.max(dim=1)
    average_score = max_similarity.mean()
    return {
        "average_score": average_score.item(),
    }


def get_tracklets(model, video_path, mask=None, target_size=None):
    """
    Extract video tracklets

    Args:
        model: CoTracker model
        video_path: Video path or a pre-loaded video tensor
        mask: Segmentation mask
        target_size: Target size (height, width); if provided, the video will be resized
    """
    if isinstance(video_path, str):
        video = read_video_from_path(video_path)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    else:
        video = video_path
    
    if target_size is not None:
        target_height, target_width = target_size
        B, T, C, H, W = video.shape
        video = video.reshape(B * T, C, H, W)
        video = torch.nn.functional.interpolate(
            video,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
        video = video.reshape(B, T, C, target_height, target_width)
    
    video = video.cuda()
    
    if mask is not None:
        mask = mask.cuda()
    
    pred_tracks_small, pred_visibility_small = model(video, grid_size=55, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c")
    return pred_tracks_small


def load_mask(mask_path):
    """Load mask and convert to bounding box"""
    segm_mask = np.array(Image.open(mask_path))
    segm_mask = torch.tensor(segm_mask).float()
    if segm_mask.max() > 1:
        segm_mask = segm_mask / 255
    
    box_mask = torch.zeros_like(segm_mask)
    nonzero = segm_mask.nonzero()
    if len(nonzero) > 0:
        minx = nonzero[:, 0].min()
        maxx = nonzero[:, 0].max()
        miny = nonzero[:, 1].min()
        maxy = nonzero[:, 1].max()
        box_mask[minx:maxx, miny:maxy] = 1
    box_mask = box_mask[None, None]
    return box_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, 
                       default="eval/motion_fidelity/configs/motion_fidelity_score_config_multi.yaml")
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)

    model = CoTrackerPredictor(checkpoint=config.cotracker_model_path)
    model = model.cuda()

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
            header = next(reader)  # Read the header row
            
            # Get the required column indices
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
                
                # Check if the video file exists
                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue
                
                print(f"Processing: {video_path}")

                # Read original video and original video mask
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
                all_similarity_matrices = []
                
                for idx, (original_video_path, original_mask_path, edit_mask_path) in enumerate(
                    zip(original_video_paths, original_mask_paths, edit_mask_paths)
                ):
                    print(f"  Processing object {idx+1}/{len(original_video_paths)}")
                    
                    if config.use_mask:
                        original_mask_file = os.path.join(original_mask_path, "0.png")
                        if os.path.exists(original_mask_file):
                            original_mask = load_mask(original_mask_file)
                            original_mask = torch.nn.functional.interpolate(
                                original_mask, 
                                size=(edit_height, edit_width), 
                                mode='nearest'
                            )
                        else:
                            print(f"  Warning: Original mask not found at {original_mask_file}, skipping mask")
                            original_mask = None
                        
                        edit_mask_file = os.path.join(edit_mask_path, "0.png")
                        if os.path.exists(edit_mask_file):
                            edit_mask = load_mask(edit_mask_file)
                            edit_mask = torch.nn.functional.interpolate(
                                edit_mask,
                                size=(edit_height, edit_width),
                                mode='nearest'
                            )
                        else:
                            print(f"  Warning: Edit mask not found at {edit_mask_file}, skipping mask")
                            edit_mask = None
                    else:
                        original_mask = None
                        edit_mask = None
                    
                    try:
                        original_tracklets = get_tracklets(
                            model, 
                            original_video_path, 
                            mask=original_mask,
                            target_size=(edit_height, edit_width)
                        )
                        
                        edit_tracklets = get_tracklets(
                            model, 
                            edit_video_path, 
                            mask=edit_mask
                        )
                        
                        similarity_matrix = get_similarity_matrix(edit_tracklets, original_tracklets)
                        similarity_scores_dict = get_score(similarity_matrix)
                        
                        all_scores.append(similarity_scores_dict["average_score"])
                        all_similarity_matrices.append(similarity_matrix.cpu())
                        
                        print(f"    Object {idx+1} score: {similarity_scores_dict['average_score']:.4f}")
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
                        'motion_fidelity': average_score
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
            f.write(f"\nOverall average motion fidelity score: {overall_avg_score:.4f}\n")
            print(f"\nOverall average motion fidelity score: {overall_avg_score:.4f}")
        else:
            f.write("\nNo videos processed.\n")
            print("\nNo videos processed.")
    
    # Write individual video results to result.csv
    if all_video_results:
        with open(result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_name', 'video_path', 'num_objects', 'individual_scores', 'motion_fidelity']
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