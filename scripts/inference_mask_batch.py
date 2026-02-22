import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
import sys
import pandas as pd
import time  # Add time module
sys.path.append(os.getcwd())

from utils import get_gt_img, get_gt_masks, compute_prompt_embeddings
from common_inference import save_output, get_motion_token_indices, generate_video

import torch
from PIL import Image
import numpy as np

# Modify command line arguments section
def get_args():
    parser = argparse.ArgumentParser(description="Inference arguments of FlexiMMT.")
    parser.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        default="../CogVideoX-5b-I2V",
        help="Path to CogVideoX-5b-I2V weights"
    )
    parser.add_argument(
        "--csv_dir", 
        type=str, 
        default="benchmark_new/captions_inf_test/val_image.csv",
        help="csv_dir"
    )
    parser.add_argument(
        "--ckpt_dir", 
        type=str, 
        default="exp_outputs_mask",
        help="ckpt_dir"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="outputs_mask_seed42",
        help="Output path for generated videos"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Output path for generated videos"
    )
    
    return parser.parse_args()

# Modify main function
if __name__ == "__main__":
    args = get_args()

    csv_files = args.csv_dir.split("+")

    print(f"Found {len(csv_files)} CSV file(s)")

    # Add total time statistics
    total_start_time = time.time()
    total_videos_processed = 0
    total_videos_success = 0
    total_videos_failed = 0

    # Iterate through all CSV files
    for csv_file in csv_files:
        print(f"\nProcessing CSV file: {csv_file}")

        df = pd.read_csv(csv_file)
        
        # Iterate through each row in CSV
        for index, row in df.iterrows():
            video_start_time = time.time()  # Start time for single video
            
            print(f"  Processing {index+1}/{len(df)} (from {csv_file})")
            
            # Get image path
            gt_img_path = row['path']
            if not os.path.exists(gt_img_path):
                print(f"Image path does not exist: {gt_img_path}, skipping")
                continue
                
            # Get sources (corresponding to emb_ckpt_path)
            sources = row['sources'].split('+')
            if not sources:
                print(f"No sources provided, skipping")
                continue
                
            # Build emb_ckpt_paths
            emb_ckpt_paths = []
            for source in sources[:5]:  # Take at most 5
                path = f"{args.ckpt_dir}/{source}/pytorch_model.pt"
                if os.path.exists(path):
                    emb_ckpt_paths.append(path)
                else:
                    print(f"Path does not exist: {path}")
            
            # Get prompt
            prompt = row['caption']
            
            # Get concept_words and motion_words
            concept_words = row['concept_words'].split('+')
            motion_words = row['motion_words'].split('+')
            
            # Ensure motion_words matches emb_ckpt_paths count
            motion_words = motion_words[:len(emb_ckpt_paths)]

            # save_name = gt_img_path.split("/")[-1] + "+" + row['sources'] + "+seed" + str(args.seed)
            save_name = gt_img_path.split("/")[-1] + "+" + row['sources']

            if os.path.exists(os.path.join(args.output_path, "videos", f"{save_name}.mp4")):
                print("Video already exists")
                continue
            
            total_videos_processed += 1
            
            # Generate image masks
            gt_img = get_gt_img(gt_img_path, 480, 720)

            # Try to find if there are predefined masks
            concept_mask_base_path = gt_img_path.replace('target_images', 'target_masks')
            concept_mask_base_path = concept_mask_base_path + "+" + row['sources']
            if os.path.exists(concept_mask_base_path):
                gt_masks = []
                for concept_word in concept_words:
                    mask_path = os.path.join(concept_mask_base_path, f"{concept_word}.png")
                    if os.path.exists(mask_path):
                        mask_img = Image.open(mask_path).convert('L')
                        mask_img = mask_img.resize((45, 30))
                        mask_array = np.array(mask_img) > 127
                        gt_masks.append(torch.from_numpy(mask_array))
                    else:
                        print(f"Warning: mask file does not exist {mask_path}")
                        gt_masks.append(torch.zeros((30, 45), dtype=torch.bool))
                
                gt_masks = torch.stack(gt_masks)
                print(f"Successfully loaded {len(concept_words)} predefined masks, shape: {gt_masks.shape}")
            else:
                print("Failed to load masks, attempting to re-extract")
                gt_masks = get_gt_masks(gt_img_path, 30, 45, concept_words, args.output_path, save_name)

            # Error log path
            error_log_path = "error_images.txt"

            try:
                video_path = generate_video(
                    prompt=prompt,
                    pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    emb_ckpt_paths=emb_ckpt_paths,
                    gt_img=gt_img,
                    gt_masks=gt_masks,
                    output_path=args.output_path,
                    guidance_scale=6.0,
                    seed=args.seed,
                    high_timesteps=None,
                    reweight_scale=None,
                    motion_words=motion_words,
                    save_name=save_name,
                )
                video_end_time = time.time()
                video_time = video_end_time - video_start_time
                total_videos_success += 1
                print(f"Successfully generated video: {video_path}")
                print(f"Single video inference time: {video_time:.2f}s ({video_time/60:.2f}min)")
            except Exception as e:
                video_end_time = time.time()
                video_time = video_end_time - video_start_time
                total_videos_failed += 1
                print(f"Error generating video: {str(e)}")
                print(f"Failed video processing time: {video_time:.2f}s")
                with open(error_log_path, "a") as f:
                    f.write(f"{save_name}\n")
            finally:
                # Clean up memory
                import gc
                del gt_img, gt_masks
                if 'emb_ckpt_paths' in locals():
                    del emb_ckpt_paths
                torch.cuda.empty_cache()  # Clear GPU cache
                gc.collect()  # Force garbage collection
                print("Memory cleaned\n")
    
    # Calculate total time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "="*60)
    print("All videos generation completed")
    print("="*60)
    print(f"Total inference time: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)")
    print(f"Total videos processed: {total_videos_processed}")
    print(f"Successfully generated: {total_videos_success}")
    print(f"Failed: {total_videos_failed}")
    if total_videos_success > 0:
        avg_time = total_time / total_videos_success
        print(f"Average inference time per video: {avg_time:.2f}s ({avg_time/60:.2f}min)")
    print("="*60)