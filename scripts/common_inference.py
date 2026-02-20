import os
import torch
from typing import List, Optional

from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from diffusers.image_processor import VaeImageProcessor

from models.my_CogVideoI2V_inf import CogVideoXTransformer3DModelIP
from models.my_pipeline_i2v_inf import CogVideoXImageToVideoPipeline
from models.autoencoder_kl_cogvideox import AutoencoderKLCogVideoX

from models.FlexiMMT_processor_mask_inf import RefNetLoRAProcessor

from utils import get_gt_img, get_gt_masks, compute_prompt_embeddings
from transformers import AutoTokenizer

import numpy as np
import random

def save_output(
    pipe,
    pipeline_args,
    device=None,
    seed=42,
    output_dir=None,
    save_name=None
):
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(device)

    # run inference
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None
    pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
    pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
    image_np = VaeImageProcessor.pt_to_numpy(pt_images) # float32 [0, 1]
    video = VaeImageProcessor.numpy_to_pil(image_np)

    save_dir = os.path.join(output_dir, "videos")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{save_name}.mp4")
    output_video_path = export_to_video(video, filename, fps=8)
    del pipe

    return output_video_path

def get_motion_token_indices(tokenizer, prompt, motion_words):
    """
    Automatically determine the token index positions and lengths of motion words in the prompt text.

    Args:
        tokenizer: CogVideoX model tokenizer
        prompt: Text prompt for video generation
        motion_words: List of motion words, e.g. ["rotate", "turning"]

    Returns:
        List of (index, length) tuples for each motion word, format: [(idx1, len1), (idx2, len2), ...]
    """
    # Tokenize the entire prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    
    results = []
    
    for motion_word in motion_words:
        # Tokenize the motion_word to get its token representation
        motion_tokens = tokenizer.encode(motion_word, add_special_tokens=False)
        motion_token_len = len(motion_tokens)

        # Search for this subsequence in the prompt tokens
        for i in range(len(tokens) - motion_token_len + 1):
            if tokens[i:i+motion_token_len] == motion_tokens:
                results.append((i, motion_token_len))
                break
                
    return results

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_video(
    gt_img: torch.Tensor,
    gt_masks: torch.Tensor,
    prompt: str,
    emb_ckpt_paths: List[str],
    pretrained_model_name_or_path: str,
    output_path: str = "outputs",
    guidance_scale: float = 6.0,
    seed: int = 42,
    high_timesteps: Optional[float] = None,
    reweight_scale: float = None,
    motion_words: List = None,
    save_name: str = None,
):
    """
    Generate video based on given prompt and target image, and save to specified path.
    """
    setup_seed(seed)
    # 1. Load pretrained CogVideoX pipeline
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    scheduler = CogVideoXDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    transformer = CogVideoXTransformer3DModelIP.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        revision=None,
        variant=None,
    )
    transformer.to(device, dtype=weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None
    )

    # Automatically locate motion words
    motion_indices = []
    if motion_words and len(motion_words) > 0:
        motion_indices = get_motion_token_indices(tokenizer, prompt, motion_words)
        
    # Limit motion_indices count to match emb_ckpt_paths
    num_embeddings = min(len(emb_ckpt_paths), len(motion_indices), 5)  # 5 max
    if num_embeddings < len(motion_indices):
        motion_indices = motion_indices[:num_embeddings]
    if num_embeddings < len(emb_ckpt_paths):
        emb_ckpt_paths = emb_ckpt_paths[:num_embeddings]
        
    print(f"Using {num_embeddings} embedding weights and motion words")
    
    # Define index ranges for each embedding (evenly distributed layers)
    total_processors = 42  # 42 processor layers total
    token_indices = []
    
    if num_embeddings > 0:
        layers_per_emb = total_processors // num_embeddings
        for i in range(num_embeddings):
            start_idx = i * layers_per_emb
            end_idx = (i + 1) * layers_per_emb if i < num_embeddings - 1 else total_processors
            # token_indices.append([start_idx, end_idx])
            token_indices.append([0, 42])
    
    # Create processor configuration
    fleximmt_procs = {}
    for i, (name, processor) in enumerate(transformer.attn_processors.items()):
        # Determine which tokens each processor uses
        use_tokens = [False] * num_embeddings
        for idx, (start_idx, end_idx) in enumerate(token_indices):
            if start_idx <= i < end_idx:
                use_tokens[idx] = True
                
        # Prepare first_attns parameter
        first_attns = [None] * num_embeddings
        for idx in range(min(num_embeddings, len(motion_indices))):
            if idx < len(motion_indices):
                motion_idx, ins_len = motion_indices[idx]
                first_attns[idx] = [motion_idx, motion_idx + ins_len]
        
        fleximmt_procs[name] = RefNetLoRAProcessor(
            stage=0 if any(use_tokens) else None,
            use_tokens=use_tokens,
            first_attns=first_attns
        )

    fleximmt_procs = fleximmt_procs.copy()
    transformer.set_attn_processor(fleximmt_procs)

    # Load all embedding weights
    ckpts = []
    for idx, path in enumerate(emb_ckpt_paths):
        if path:
            try:
                ckpt = torch.load(path, weights_only=True)
                ckpts.append(ckpt)
                print(f"Loaded motion token #{idx+1}: {path}")
            except Exception as e:
                print(f"Failed to load motion token {path}: {e}")
                ckpts.append(None)
    
    # Update processor weights
    for i, (name, proc) in enumerate(transformer.attn_processors.items()):
        for idx, (start_idx, end_idx) in enumerate(token_indices):
            if idx < len(ckpts) and ckpts[idx] is not None:
                if start_idx <= i < end_idx and hasattr(proc, f"tokens_{idx}"):
                    token_key = f"{name}.motion_inversion_tokens"
                    if token_key in ckpts[idx]:
                        getattr(proc, f"tokens_{idx}").data.copy_(ckpts[idx][token_key])
    
    print("Successfully loaded all available motion_inversion_tokens")
    
    # Create pipeline
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=transformer,
        scheduler=scheduler,
        revision=None,
        variant=None,
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # 2. Set up model on device
    pipe.to(device=device, dtype=weight_dtype)
    vae.to(device=device, dtype=weight_dtype)
    vae.enable_slicing()
    vae.enable_tiling()
    pipe.vae = vae
    
    # Inference
    # 3. Process inputs
    prompts = ["", prompt]
    text_encoder.to(device=device)
    prompt_embeds = compute_prompt_embeddings(
        tokenizer,
        text_encoder,
        prompts,
        226,
        device,
        weight_dtype,
        requires_grad=False,
    ) # [2B, L, C]
    text_encoder.to(device='cpu')
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
    
    # Sample random timesteps for each image
    b, t, c, h, w = 1, 13, 16, 480 // 8, 720 // 8
    noise = torch.randn(b, t, c, h, w, device=device)
    # Use first frame as concat image conditioning
    img_cat_latents = pipe.vae.encode(gt_img.permute(0, 2, 1, 3, 4)[:, :, :1].to(dtype=weight_dtype, device=device)).latent_dist.sample()
    img_cat_latents = img_cat_latents.permute(0, 2, 1, 3, 4) * pipe.vae.config.scaling_factor  # [B, 1, C, H, W]
    padding_shape = (b, t - 1, c, h, w)
    latent_padding = torch.zeros(padding_shape).to(device=device, dtype=weight_dtype)
    img_cat_latents = torch.cat([img_cat_latents, latent_padding], dim=1) # [B, F, C, H, W]

    # 4. Generate video frames based on prompt
    pipeline_args = {
        "image": img_cat_latents,
        "first_masks": gt_masks,
        "guidance_scale": guidance_scale,
        "use_dynamic_cfg": False,
        "height": 480,
        "width": 720,
        "latents": noise.to(dtype=weight_dtype, device=device),
        "prompt_embeds": prompt_embeds,
        "num_frames": 49,
        "high_timesteps": high_timesteps,
        "reweight_scale": reweight_scale,
        "output_dir": output_path,
        "save_name": save_name,
    }
    output_video_path = save_output(
        pipe=pipe,
        pipeline_args=pipeline_args,
        device=device,
        seed=seed,
        output_dir=output_path,
        save_name=save_name
    )
    print(f"Generated video saved to {output_video_path}")
    return output_video_path