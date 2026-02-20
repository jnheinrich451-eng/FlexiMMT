from transformers import T5EncoderModel, T5Tokenizer
from typing import List, Optional, Union, Tuple
import argparse
import torch
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import torchvision.transforms as TT
import numpy as np
import random
import os
from decord import VideoReader, cpu
from PIL import Image
from collections import OrderedDict
from diffusers import CogVideoXDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video
from diffusers.training_utils import free_memory
import torch.nn as nn
from tools.grounding_sam import grounded_segmentation
import cv2

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches


####### Parse Args >>>>>>>
def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    ## [Model information]
    parser.add_argument("--biased_loss_ratio", type=float, default=1.0, 
                            help="Biased loss ratio for FAE training.")
    parser.add_argument("--use_biased_loss", type=int, default=1, 
                            help="Whether to use biased loss for training.")
    parser.add_argument("--use_different_first_frame", action="store_true", default=False, 
                            help="Whether to use different first frame for FAE training.")
    parser.add_argument("--stage", type=int, default=None, 
                            help="Stage for FlexiMMT training.")
    parser.add_argument("--val_csv_path", type=str, default=None,
                            help="Path to validation csv file.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, 
                            help="Path to pretrained model or model identifier from huggingface.co/models.")
    
    ## [LoRA]
    parser.add_argument("--lora_weight", type=float, default=0.0, 
                            help="Lora weight for RefAdapter.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, 
                            help="Lora alpha for RefAdapter.")
    parser.add_argument("--rank", type=int, default=4, 
                            help="Lora rank for RefAdapter.")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0, 
                            help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    parser.add_argument("--revision", type=str, default=None, required=False, 
                            help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, 
                            help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16")
    parser.add_argument("--cache_dir", type=str, default=None, 
                            help="The directory where the downloaded models and datasets will be stored.")
    parser.add_argument("--ckpt_path", type=str, default=None, 
                            help="Path to the checkpoint file.")

    ## [ Dataset information ]
    parser.add_argument("--meta_file_path", type=str, default=None, 
                            help="The path to training meta data.")
    parser.add_argument("--val_meta_file_path", type=str, default=None, 
                            help="The path to validation meta data.")
    parser.add_argument("--instance_data_root", type=str, default=None, 
                            help="The training video folder.")
    parser.add_argument("--video_column", type=str, default="video", 
                            help="The column of the dataset containing videos.")
    parser.add_argument("--caption_column", type=str, default="text", 
                            help="The column of the dataset containing the instance prompt for each video. Or, the name of the file in `--instance_data_root` folder containing the line-separated instance prompts.")
    parser.add_argument("--id_token", type=str, default=None, 
                            help="Identifier token appended to the start of each prompt if provided.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, 
                            help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")

    ## [Validation information]
    parser.add_argument("--guidance_scale", type=float, default=6, 
                            help="The guidance scale to use while sampling validation videos.")
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False, 
                            help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.")
    parser.add_argument("--num_validation_videos", type=int, default=1, 
                            help="Number of validation videos to sample.")
    parser.add_argument("--emptytxt", action="store_true", default=False, 
                            help="Whether to use empty text for validation.")

    ## [Training information]
    parser.add_argument("--seed", type=int, default=None, 
                            help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                            help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="cogvideox-i2v-lora", 
                            help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--height", type=int, default=480, 
                            help="All input videos are resized to this height.")
    parser.add_argument("--width", type=int, default=720, 
                            help="All input videos are resized to this width.")
    parser.add_argument("--video_reshape_mode", type=str, default="center", 
                            help="All input videos are reshaped to this mode. Choose between ['center', 'random', 'none']")
    parser.add_argument("--fps", type=int, default=8, 
                            help="All input videos will be used at this FPS.")
    parser.add_argument("--max_num_frames", type=int, default=49, 
                            help="All input videos will be truncated to these many frames.")
    parser.add_argument("--skip_frames_start", type=int, default=0, 
                            help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.")
    parser.add_argument("--skip_frames_end", type=int, default=0, 
                            help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.")
    parser.add_argument("--random_flip", action="store_true", 
                            help="whether to randomly flip videos horizontally")
    parser.add_argument("--train_batch_size", type=int, default=4, 
                            help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                            help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, 
                            help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, 
                            help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--validating_steps", type=int, default=50, 
                            help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=10, 
                            help=("Max number of checkpoints to store."))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                            help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                            help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                            help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=True, 
                            help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", 
                            help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, 
                            help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, 
                            help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, 
                            help="Power factor of the polynomial scheduler.")
    parser.add_argument("--enable_slicing", action="store_true", default=False, 
                            help="Whether or not to use VAE slicing for saving memory.")
    parser.add_argument("--enable_tiling", action="store_true", default=False, 
                            help="Whether or not to use VAE tiling for saving memory.")
    parser.add_argument("--noised_image_dropout", type=float, default=0.05, 
                            help="Image condition dropout probability.")

    ## [Optimizer]
    parser.add_argument("--optimizer", type=lambda s: s.lower(), default="adam", choices=["adam", "adamw", "prodigy"], 
                            help=("The optimizer type to use."))
    parser.add_argument("--use_8bit_adam", action="store_true", 
                            help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
                            help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, 
                            help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--prodigy_beta3", type=float, default=None, 
                            help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.")
    parser.add_argument("--prodigy_decouple", action="store_true", 
                            help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, 
                            help="Weight decay to use for unet params")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, 
                            help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
                            help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", 
                            help="Turn on Adam's bias correction.")
    parser.add_argument("--prodigy_safeguard_warmup", action="store_true", 
                            help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.")

    ## [Other information]
    parser.add_argument("--tracker_name", type=str, default=None, 
                            help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", 
                            help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, 
                            help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, 
                            help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type=str, default="logs", 
                            help="Directory where logs are stored.")
    parser.add_argument("--allow_tf32", action="store_true", 
                            help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default=None, 
                            help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--max_text_seq_length", type=int, default=226, 
                            help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).")
    parser.add_argument("--nccl_timeout", type=int, default=6000, 
                            help="NCCL backend timeout in seconds.")
    
    parser.add_argument("--prompt", type=bool,
                            help="prompt")
    
    parser.add_argument(
        "--concept_words", 
        type=str, 
        default="person",
        help="Concept word list for controlling subject objects in video generation. Example: --concept_words person+dog"
    )
    parser.add_argument(
        "--motion_words", 
        type=str, 
        default="turning",
        help="Motion word list for controlling motion features in video. Example: --motion_words rotating+turning"
    )

    parser.add_argument("--use_mask", type=str, help="Whether to use mask")

    return parser.parse_args()


####### Prompt Embedding >>>>>>>
def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if text_input_ids is None:
        b = len(prompt)
    else:
        b = text_input_ids.shape[0]

    if tokenizer is not None and text_input_ids is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(b * num_videos_per_prompt, seq_len, -1)
    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False, token_ids=None
):
    r"""
    Compute the prompt embeddings for the given prompt [str, token_ids].

    Parameters:
        tokenizer (`T5Tokenizer`):
            The T5 tokenizer to use.
        text_encoder (`T5EncoderModel`):
            The T5 text encoder to use.
        prompt (`str`):
            The prompt to compute the embeddings for.
        max_sequence_length (`int`):
            The maximum sequence length of the input text embeddings.
        device (`torch.device`):
            The device to use.
        dtype (`torch.dtype`):
            The dtype to use.
        requires_grad (`bool`, defaults to `False`):
            Whether to require gradients for the prompt embeddings.
        token_ids (`torch.Tensor`):
            The token ids to compute the embeddings for.
    Returns:
        prompt_embeds (`torch.Tensor`):
            The prompt embeddings.
    """
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            text_input_ids=token_ids,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                text_input_ids=token_ids,
            )
    return prompt_embeds



####### ROPE >>>>>>>
def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


####### Resize >>>>>>>
def resize_wo_crop(arr, image_size):
    arr = resize(
        arr,
        size=[image_size[0], image_size[1]],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return arr


def resize_for_rectangle_crop(arr, height, width, reshape_mode="center"):
    r"""
    Resize the input array to the given height and width.
    """
    image_size = height, width
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    
    return arr


####### Load video or image >>>>>>>
def get_gt_img(gt_img_path, height, width):
    r"""
    Get the target image. Returns a tensor of shape [B, 1, C, H, W].
    """
    if isinstance(gt_img_path, str):
        gt_img = Image.open(gt_img_path)
    # Load as tensor [T, C, H, W]
    gt_img = gt_img.convert("RGB")
    gt_img = np.array(gt_img, dtype=np.uint8)
    gt_img = torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0)
    # Resize with center crop
    gt_img = resize_wo_crop(gt_img, [height, width])
    # Convert to correct format and normalize to [-1, 1] range
    gt_img = gt_img.unsqueeze(0).float() / 127.5 - 1.0

    return gt_img

def get_gt_img_origin(gt_img_path, height, width):
    r"""
    Get the target image. Returns a tensor of shape [B, 1, C, H, W].
    """
    if isinstance(gt_img_path, str):
        gt_img = Image.open(gt_img_path)
    # Load as tensor [T, C, H, W]
    gt_img = gt_img.convert("RGB")
    gt_img = np.array(gt_img, dtype=np.uint8)
    gt_img = torch.tensor(gt_img).permute(2, 0, 1).unsqueeze(0)
    # Resize with center crop
    gt_img = resize_wo_crop(gt_img, [height, width])
    # Convert to correct format and normalize to [0, 1] range
    gt_img = gt_img.unsqueeze(0).float() / 255

    return gt_img

def get_gt_masks(gt_img_path, target_height, target_width, labels, output_path=None, save_name=None):

    image_array, detections = grounded_segmentation(
        image=gt_img_path,
        labels=labels,
        threshold=0.45,
        polygon_refinement=True,
    )

    # Get original image dimensions
    if isinstance(image_array, np.ndarray):
        original_height, original_width = image_array.shape[:2]
    else:
        # If image_array is not a numpy array, try reading from file
        img = cv2.imread(gt_img_path)
        original_height, original_width = img.shape[:2]

    # Create a dictionary sorted by input labels
    sorted_detections = {label: [] for label in labels}
    
    # Group detection results by label
    for detection in detections:
        label = detection.label.rstrip('.')
        if label in sorted_detections:
            sorted_detections[label].append(detection)
    
    # Reorganize detection results in input label order
    ordered_detections = []
    ordered_labels = []
    for label in labels:
        if sorted_detections[label]:
            # If a label has multiple detections, add all of them
            for det in sorted_detections[label]:
                ordered_detections.append(det)
                ordered_labels.append(label)
    
    num_concepts = len(ordered_detections)
    
    # Initialize result array
    result = np.zeros((num_concepts, target_height, target_width), dtype=bool)
    boxes = []
    
    for i, detection in enumerate(ordered_detections):
        # Get current mask
        mask = detection.mask
        box = detection.box
        
        # Collect bounding box info
        boxes.append((box.xmin, box.ymin, box.xmax, box.ymax))
        
        # Resize mask if shape differs from target
        if mask.shape[0] != target_height or mask.shape[1] != target_width:
            mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to boolean
        mask_bool = mask.astype(bool)
        
        # Store in result array
        result[i] = mask_bool
    
    if output_path != None and save_name != None:
        visualize_stacked_masks(result, target_height, target_width, os.path.join(output_path, save_name),
                            labels=ordered_labels, boxes=boxes, original_dimensions=(original_height, original_width))
    
    return torch.from_numpy(result)

def visualize_stacked_masks(stacked_masks, target_height, target_width, output_path, name="sam_mask.png", labels=None, boxes=None, original_dimensions=None):
    """
    Visualize stacked boolean masks and bounding boxes, then save the visualization.
    
    Args:
        stacked_masks: Boolean array of shape [num_concepts, height*width]
        target_height: Height for reshaping masks
        target_width: Width for reshaping masks
        output_path: Path to save the visualization
        labels: List of labels for each mask (optional)
        boxes: List of bounding boxes in format (xmin, ymin, xmax, ymax) (optional)
        original_dimensions: Tuple (original_height, original_width) to scale boxes if needed
    """

    # Ensure save path exists
    if output_path is None:
        output_path = os.getcwd()
    os.makedirs(output_path, exist_ok=True)

    num_concepts = stacked_masks.shape[0]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_concepts + 1, figsize=(4 * (num_concepts + 1), 4))
    if num_concepts == 0:
        axes = [axes]
    
    # Generate distinct colors for each mask
    colors = list(mcolors.TABLEAU_COLORS.values())
    if num_concepts > len(colors):
        # Add more colors if needed
        colors.extend(list(mcolors.CSS4_COLORS.values())[0:num_concepts-len(colors)])
    
    # Create an overlay image for all masks combined
    combined_img = np.zeros((target_height, target_width, 3), dtype=np.float32)
    
    # Scale bounding boxes if needed
    scaled_boxes = []
    if boxes and original_dimensions:
        orig_h, orig_w = original_dimensions
        scale_x = target_width / orig_w
        scale_y = target_height / orig_h
        
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            scaled_box = (
                int(xmin * scale_x),
                int(ymin * scale_y),
                int(xmax * scale_x),
                int(ymax * scale_y)
            )
            scaled_boxes.append(scaled_box)
    else:
        scaled_boxes = boxes
    
    # Visualize each mask separately
    for i in range(num_concepts):
        # Reshape the mask back to 2D
        mask = stacked_masks[i]
        
        # Display the individual mask
        axes[i].imshow(mask, cmap='gray')
        title = f"Mask {i+1}"
        if labels and i < len(labels):
            title = f"{labels[i]}"
        axes[i].set_title(title)
        
        # Add bounding box if available
        if scaled_boxes and i < len(scaled_boxes):
            box = scaled_boxes[i]
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1], 
                linewidth=2, 
                edgecolor=colors[i % len(colors)], 
                facecolor='none'
            )
            axes[i].add_patch(rect)
        
        axes[i].axis('off')
        
        # Add this mask to the combined image with a unique color
        color_mask = np.zeros((target_height, target_width, 3), dtype=np.float32)
        color = np.array(mcolors.to_rgb(colors[i % len(colors)]))
        for c in range(3):
            color_mask[:, :, c] = mask * color[c]
        
        # Add to combined image with some transparency for overlapping regions
        combined_img += color_mask * 0.7
    
    # Clip values to valid range [0, 1]
    combined_img = np.clip(combined_img, 0, 1)
    
    # Display the combined visualization
    axes[-1].imshow(combined_img)
    axes[-1].set_title("Combined Masks")
    
    # Add all bounding boxes to the combined visualization
    if scaled_boxes:
        for i, box in enumerate(scaled_boxes):
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1], 
                linewidth=2, 
                edgecolor=colors[i % len(colors)], 
                facecolor='none'
            )
            axes[-1].add_patch(rect)
    
    axes[-1].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, name), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_path, name)}")


####### Save/Load trainable parameters >>>>>>>
def save_trainable_state_dict_wo_ds(unwrapped_model, save_path, sd=None):
    r"""
    Save the trainable parameters.
    """
    os.makedirs(save_path, exist_ok=True)
    transformer_ipa_layers_to_save = {}
    for n, p in unwrapped_model.named_parameters():
        if p.requires_grad:
            transformer_ipa_layers_to_save[n] = p
    transformer_ipa_layers_to_save = sd if sd is not None else transformer_ipa_layers_to_save
    trainable_params = [p for p in transformer_ipa_layers_to_save.values() if p.requires_grad]
    print('Saving', sum([p.numel() for p in trainable_params]) / 1000000, 'M parameters')
    sd = unwrapped_model.state_dict()
    trainable_state_dict = {k: v for k, v in sd.items() if k in transformer_ipa_layers_to_save}
    torch.save(OrderedDict(trainable_state_dict), f"{save_path}/pytorch_model.pt")

def save_motion_token_transformer_wo_ds(motion_token, unwrapped_model, save_path, sd=None):
    r"""
    Save the trainable parameters.
    """
    os.makedirs(save_path, exist_ok=True)

    transformer_ipa_layers_to_save = {}
    for n, p in unwrapped_model.named_parameters():
        if p.requires_grad:
            transformer_ipa_layers_to_save[n] = p
    transformer_ipa_layers_to_save = sd if sd is not None else transformer_ipa_layers_to_save
    trainable_params = [p for p in transformer_ipa_layers_to_save.values() if p.requires_grad]
    print('Saving', sum([p.numel() for p in trainable_params]) / 1000000, 'M parameters')
    sd = unwrapped_model.state_dict()
    trainable_state_dict = {k: v for k, v in sd.items() if k in transformer_ipa_layers_to_save}
    torch.save(OrderedDict(trainable_state_dict), f"{save_path}/pytorch_model.pt")

    motion_token_data = motion_token.data.clone().cpu()
    torch.save(motion_token_data,  f"{save_path}/motion_token.pt")

def save_motion_token_wo_ds(motion_token, save_path, sd=None):
    r"""
    Save the trainable parameters.
    """
    os.makedirs(save_path, exist_ok=True)
    motion_token_data = motion_token.data.clone().cpu()
    torch.save(motion_token_data,  f"{save_path}/pytorch_model.pt")


def load_trainable_state_dict_wo_ds(unwrapped_model, load_path):
    r"""
    Load the trainable parameters.
    """
    if not os.path.exists(load_path):
        print(f"Checkpoint '{load_path}' does not exist. Starting a new training run.")
        return
    unwrapped_model.load_state_dict(torch.load(load_path), strict=False)
    print(f"Loaded {load_path}")

def load_text_encoder(text_encoder, load_path, modifier_token_ids=None):
    r"""
    Load the trainable parameters.
    Only update embeddings for specific tokens, keep others unchanged.

    Args:
        text_encoder: Text encoder model
        load_path: Path to load weights from
        modifier_token_ids: List of token IDs to update, if None update all
    """
    if not os.path.exists(load_path):
        print(f"Checkpoint '{load_path}' does not exist. Starting a new training run.")
        return {}
    
    loaded_data = torch.load(load_path)

    # Get current embedding layer for structure info
    current_embeddings = text_encoder.get_input_embeddings()

    # Record original embedding attributes
    device = current_embeddings.weight.device
    dtype = current_embeddings.weight.dtype

    # Prepare return dict to store loaded token embeddings
    loaded_token_embeddings = {}

    # Load token embeddings from saved state dict
    for key, value in loaded_data.items():
        if key.startswith("shared.weight_"):
            token_id = key.split("_")[1]
            loaded_token_embeddings[token_id] = value.to(device=device, dtype=dtype)
    
    return loaded_token_embeddings


def load_motion_token_wo_ds(motion_token, save_path, sd=None):
    r"""
    Save the trainable parameters.
    """
    os.makedirs(save_path, exist_ok=True)
    torch.save(motion_token,  f"{save_path}/pytorch_model.pt")


####### Evaluation >>>>>>>
def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    global_step = 0,
    is_final_validation: bool = False,
    gt_img=None,
    caption=None,
):
    r"""
    Log the validation results.

    Parameters:
        pipe (`CogVideoXImageToVideoPipeline`):
            The I2V pipeline.
        global_step (`int`):
            The global step.
        is_final_validation (`bool`):
            Whether to log the final validation results.
        gt_img (`torch.Tensor` , [B, T, C, H, W]):
            The target image.
    """
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    if accelerator.is_main_process:
        for _ in range(args.num_validation_videos):
            pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
            pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

            image_np = VaeImageProcessor.pt_to_numpy(pt_images) # float32 [0, 1]
            if gt_img is not None:
                gt_img = gt_img[0].permute(0, 2, 3, 1)
                gt_imgs_np = (np.array(gt_img) + 1.0) / 2.0
                if gt_imgs_np.shape[0] != image_np.shape[0]:
                    gt_imgs_np = np.tile(gt_imgs_np, (image_np.shape[0], 1, 1, 1))
                image_np = np.concatenate([gt_imgs_np, image_np], axis=2)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)

            videos.append(image_pil)

    phase_name = "test" if is_final_validation else "validation"
    video_filenames = []
    for i, video in enumerate(videos):
        cap_len = min(len(caption), 100)
        caption = caption[:cap_len]
        save_dir = os.path.join(args.output_dir, "videos", f"step{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        idx = 0
        filename = os.path.join(save_dir, f"inf_{caption}_{idx}.mp4")
        while os.path.exists(filename):
            idx += 1
            filename = os.path.join(save_dir, f"inf_{caption}_{idx}.mp4")
        export_to_video(video, filename, fps=8)
        video_filenames.append(filename)

    del pipe
    free_memory()

    return videos


####### Get Optimizer >>>>>>>
def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        print(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        print(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            print(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer