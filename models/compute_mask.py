import os
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm
import time

import gc
import cv2
import torch
from torch.nn import functional as F
from PIL import Image
from einops import rearrange

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

@torch.no_grad()
def compute_mask(first_masks, intermediate_feature, height, width, num_frames=13, min_size_threshold=64):
    """
    Args:
        first_masks: torch.Tensor, shape [height, width] - mask of the first frame
        intermediate_feature: torch.Tensor, shape [1, num_frames * height * width, channel] - intermediate features
        width: width
        height: height
        num_frames: number of frames, default 13
        min_size_threshold: minimum size threshold, no interpolation scaling below this value
    Returns:
        torch.Tensor: segmentation results for all frames [num_frames, height, width]
    """
    # Get device information
    device = first_masks.device
    
    # Reshape intermediate_feature
    # From [1, num_frames * height * width, channel] to [num_frames, channel, height, width]
    total_pixels, channel = intermediate_feature.shape
    
    # Reshape features
    intermediate_feature_reshaped = intermediate_feature.view(num_frames, height, width, channel)
    intermediate_feature_reshaped = intermediate_feature_reshaped.permute(0, 3, 1, 2)  # [num_frames, channel, height, width]

    # visualize_features(intermediate_feature_reshaped)
    # Process first frame mask - decide whether interpolation is needed based on size
    max_dim = max(height, width)
    
    # Ensure first_masks has correct dimensions [height, width] -> [1, 1, height, width]
    if first_masks.dim() == 2:
        first_masks_input = first_masks.unsqueeze(0).unsqueeze(0)
    else:
        first_masks_input = first_masks
    
    mask_edit_reshape = first_masks_input.squeeze(0).squeeze(0).to(device=device, dtype=intermediate_feature.dtype)
    
    # Select features
    inter_feature_2 = intermediate_feature_reshaped  # [num_frames, channel, height, width]
        
    # Convert first frame mask to one-hot encoding
    first_seg = to_one_hot(mask_edit_reshape.unsqueeze(0))  # Add batch dimension
    
    # Queue to store the last n frames
    # n_last_frames = int(num_frames / 3)
    n_last_frames = 2
    que = queue.Queue(n_last_frames)
    
    # Extract first frame features
    frame1_feat = read_feature(inter_feature_2, 0).T  # dim x h*w
    
    frame_tar_segs = []
    frame_tar_segs.append(first_masks)  # Directly use the input first_masks
    all_seg_np = []
    for cnt in range(1, num_frames):
        # Use first frame segmentation and results from the last n frames
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg.squeeze(0)[1:].flatten(1).to(device=device, dtype=intermediate_feature.dtype)] + [pair[1].to(device=device, dtype=intermediate_feature.dtype) for pair in list(que.queue)]
        mask_binary, seg_sample, feat_tar, seg_np = label_propagation_nearby(inter_feature_2, used_frame_feats, used_segs, index=cnt)
        
        # Collect results
        all_seg_np.append(seg_np)

        # Remove the oldest frame (if needed)
        if que.qsize() == n_last_frames:
            que.get()
        
        # Put current result into the queue
        seg = copy.deepcopy(seg_sample)
        que.put([feat_tar, seg])

        frame_tar_segs.append(mask_binary.to(device))

    return torch.stack(frame_tar_segs), all_seg_np

def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask

def read_feature(data, frame_index, return_h_w=False):
    """Extract one frame feature everytime."""
    dim, h, w = data[frame_index].shape
    data = rearrange(data[frame_index], "c h w->(h w) c") # hw,c

    if return_h_w:
            return data, h, w
    return data

def label_propagation_nearby(inter_feature_2, list_frame_feats, list_segs, index=None):
    """
    Improved label propagation function - uses original mask (no dilation) for source frames, uses dilated mask for target frame
    """
    # Parameter settings
    # temperature = 0.015
    temperature = 0.013
    boundary_radius = 2
    
    # Memory cleanup and feature extraction
    gc.collect()
    torch.cuda.empty_cache()
    feat_tar, h, w = read_feature(inter_feature_2, index, return_h_w=True)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Prepare data
    segs = torch.cat(list_segs, dim=-1) # 1 is foreground
    return_feat_tar = feat_tar.T
    
    # Extract source frame foreground regions - no dilation
    src_masks = []
    for i, seg in enumerate(list_segs):
        # Extract foreground mask
        seg_2d = seg[0, :].view(h, w)
        seg_np = seg_2d.detach().float().cpu().numpy().astype(np.float32)
        mask_binary = (seg_np > 0.5).astype(np.uint8)
        
        # Directly use the original mask, no dilation
        src_mask = torch.from_numpy(mask_binary.flatten()).to(seg.device)
        src_masks.append(src_mask)
    
    # Get source frame and target frame masks
    # Source frame mask
    src_masks_tensor = torch.stack(src_masks)
    src_masks_2d = src_masks_tensor.reshape(-1)
    # Target frame mask needs logical_or with the first frame
    target_mask = src_masks[-1] > 0
    target_mask_2d = target_mask.reshape(h, w).cpu().numpy().astype(np.uint8)
    
    target_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 (boundary_radius*2+1, boundary_radius*2+1))
    target_dilate_mask = cv2.dilate(target_mask_2d, target_dilate_kernel, iterations=1)
    target_dilate_mask_2d = torch.from_numpy(target_dilate_mask.flatten()).to(feat_tar.device)
    
    feat_sources = torch.cat(list_frame_feats, dim=-1)
    

    # Apply masks before computing affinity matrix
    src_mask_soft = torch.where(src_masks_2d > 0, 
                            torch.ones_like(src_masks_2d), 
                            torch.ones_like(src_masks_2d) * 0.01)
    src_mask_expanded = src_mask_soft.unsqueeze(0)
    feat_sources = feat_sources * src_mask_expanded.to(dtype=feat_sources.dtype)

    # Feature normalization
    feat_tar_norm = F.normalize(feat_tar, dim=1, p=2).squeeze(0)
    feat_sources = F.normalize(feat_sources, dim=0, p=2)

    # Compute affinity matrix
    aff = torch.exp(torch.mm(feat_tar_norm, feat_sources) / temperature)
    aff = aff.transpose(1, 0)  # (num_context*h*w) x (h*w)
    
    # TopK filtering
    tk_val, _ = torch.topk(aff, dim=0, k=min(15, aff.shape[0]))
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0
    
    # Normalization
    aff_sum = torch.sum(aff, dim=0, keepdim=True)
    valid_cols = aff_sum.squeeze(0) > 0
    if valid_cols.any():
        aff[:, valid_cols] = aff[:, valid_cols] / aff_sum[:, valid_cols]
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Generate segmentation results
    seg_tar = torch.mm(segs, aff)
    
    # Post-processing
    seg_2d = seg_tar.squeeze(0).reshape(h, w)
    
    # Compute mean and standard deviation within the mask region
    mask_region = torch.from_numpy(target_mask_2d).to(feat_tar.device) > 0
    if mask_region.sum() > 0:  # Ensure the mask region is not empty
        values_in_mask = seg_2d[mask_region]
        # Sort values to remove the smallest and largest 10%
        sorted_values, _ = torch.sort(values_in_mask)
        num_values = sorted_values.size(0)
        # Calculate the number of smallest and largest values to skip (10% each)
        skip_count = int(num_values * 0.1)
        # If there are enough values to skip
        if skip_count > 0 and num_values > skip_count * 2:
            # Compute mean after removing the smallest and largest 10%
            filtered_values = sorted_values[skip_count:-skip_count]
            mean_value = filtered_values.mean().item()
        else:
            # If too few samples, use all values to compute mean
            mean_value = values_in_mask.mean().item()
        # Dynamic threshold calculation - use mean as base, adjust coefficient as needed
        dynamic_threshold = mean_value * 0.98 # Or other formulas, e.g. mean_value - 0.5 * std_value
        # Set threshold lower bound to avoid threshold being too low
        dynamic_threshold = max(0.3, dynamic_threshold)
    else:
        # If mask region is empty, use default threshold
        dynamic_threshold = 0.5

    seg_np = seg_2d.detach().float().cpu().numpy().astype(np.float32)
    mask_binary = seg_2d > dynamic_threshold

    return mask_binary, seg_tar, return_feat_tar, seg_np

def visualize_features(features, save_dir='test_pre', prefix='frame'):
    """
    Visualize intermediate features by averaging across channels.
    
    Args:
        features: torch.Tensor, shape [num_frames, channel, height, width]
        save_dir: Directory to save visualizations
        prefix: Prefix for saved files
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get feature dimensions
    num_frames, channels, height, width = features.shape
    
    # Loop through frames
    for i in range(num_frames):
        # Average across channels
        feature_avg = features[i].sum(dim=0).float().cpu().numpy()
        
        # Normalize for better visualization
        feature_min = feature_avg.min()
        feature_max = feature_avg.max()
        if feature_max > feature_min:
            feature_avg = (feature_avg - feature_min) / (feature_max - feature_min)
        
        # Create figure
        plt.figure(figsize=(8, 8))
        plt.imshow(feature_avg, cmap='viridis')
        plt.colorbar(label='Average Activation')
        plt.title(f'Frame {i} Feature Map (Channel Average)')
        plt.axis('on')
        
        # Save figure
        save_path = os.path.join(save_dir, f'{prefix}_{i:03d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
    print(f"Saved {num_frames} feature visualizations to {save_dir}")

def save_segmentation_heatmap(seg_np, save_path, colormap='jet', alpha=0.7, dpi=150, 
                             add_colorbar=True, title="hotmap"):
    """
    Save segmentation probability map as a heatmap

    Args:
        seg_np: numpy array, shape (height, width), segmentation probability with values in [0,1]
        save_path: string, path to save the heatmap
        colormap: string, color mapping for the heatmap, default 'jet'
        alpha: float, transparency of the heatmap, default 0.7
        dpi: int, image resolution, default 150
        add_colorbar: bool, whether to add a colorbar, default True
        title: string, image title

    Returns:
        None, saves the image directly to the specified path
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    import os
    
    # Ensure directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    # Ensure input is a 2D array
    assert len(seg_np.shape) == 2, f"Input should be a 2D array, current shape: {seg_np.shape}"
    
    # Create figure
    im = plt.imshow(seg_np, cmap=colormap, interpolation='bilinear', vmin=0, vmax=1)
    
    # Add colorbar
    if add_colorbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('prob', fontsize=12)
        
        # Add key value labels on the colorbar
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0 (bg)', '0.25', '0.5', '0.75', '1.0 (forg)'])
    
    # Set title and axes
    plt.title(title, fontsize=14)
    plt.axis('on')  # Show axes
    plt.grid(False)  # Hide grid
    
    
    # Save image
    plt.tight_layout()
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Heatmap saved to: {save_path}")
    except Exception as e:
        print(f"Error saving heatmap: {e}")
    finally:
        plt.close()

def visualize_binary_mask(combined_mask, title="Binary Mask Visualization", 
                         save_path=None, show_stats=True):
    """
    Visualize binary mask

    Args:
        combined_mask: numpy array with shape (height, width), dtype uint8, values 0 or 1
        title: image title
        save_path: save path, if None then do not save
        show_stats: whether to show statistics
    """
    # Validate input
    assert combined_mask.dtype == np.uint8, f"Expected uint8, got {combined_mask.dtype}"
    unique_vals = np.unique(combined_mask)
    assert set(unique_vals).issubset({0, 1}), f"Expected only 0 and 1, got {unique_vals}"
    
    height, width = combined_mask.shape
    
    # Create figure
    plt.figure(figsize=(10, 8))

    # Use black-and-white colormap, 0 is black, 1 is white
    # Or use a custom colormap for more obvious contrast
    colors = ['black', 'white']  # 0->black, 1->white
    cmap = ListedColormap(colors)
    
    im = plt.imshow(combined_mask, cmap=cmap, interpolation='nearest')
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Image saved to: {save_path}")
    
def visualize_seg_tar_grid(seg_tar, height, width, max_samples=9, 
                                        title="Segmentation Grid", cmap='tab20', 
                                        save_path=None, show_global_colorbar=True):
    """
    Visualize segmentation results of multiple batches in a grid, with a global colorbar

    Args:
        seg_tar: torch.bfloat16 tensor with shape [batch, height*width]
        height: image height
        width: image width
        max_samples: maximum number of samples to display
        title: overall title
        cmap: color mapping
        save_path: save path
        show_global_colorbar: whether to show a global colorbar
    """
    batch_size = seg_tar.shape[0]
    n_samples = min(batch_size, max_samples)
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    
    # If showing colorbar, adjust image size
    if show_global_colorbar:
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols + 1, 3*rows))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    fig.suptitle(title, fontsize=16)
    
    # Handle axes shape
    if n_samples == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Calculate global value range for unified colorbar
    all_data = seg_tar.cpu().float().numpy()
    global_min, global_max = all_data.min(), all_data.max()
    unique_global_values = np.unique(all_data)
    
    images = []
    for i in range(n_samples):
        seg_data = seg_tar[i].cpu().float().numpy().reshape(height, width)
        
        ax = axes[i] if n_samples > 1 else axes[0]
        im = ax.imshow(seg_data, cmap=cmap, interpolation='nearest', 
                      vmin=global_min, vmax=global_max)
        ax.set_title(f'Batch {i}', fontsize=10)
        ax.axis('off')
        images.append(im)
    
    # Hide extra subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Grid image saved to: {save_path}")

def label_propagation(inter_feature_2, list_frame_feats, list_segs, index=None):
    temperature = 0.05
    sample_ratio = 2
    # ----------------------------------------------------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------------------------
    ## we only need to extract feature of the target frame
    feat_tar, h, w = read_feature(inter_feature_2, index, return_h_w=True)
    # ----------------------------------------------------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------------------------
    # load mask
    segs = torch.cat(list_segs, dim=-1) # C x nmb_context*h*w
    C, _ = segs.shape
    nmb_context = len(list_segs)
    return_feat_tar = feat_tar.T # dim x h*w
    ncontext = len(list_frame_feats)

    feat_tar = F.normalize(feat_tar, dim=1, p=2).squeeze(0)
    feat_sources = torch.cat(list_frame_feats, dim=-1) # nmb_context x dim x h*w
    feat_sources = F.normalize(feat_sources, dim=0, p=2)

    aff = torch.exp(torch.mm(feat_tar, feat_sources) / temperature) # nmb_context x h*w (tar: query) x h*w (source: keys)
    # nmb_context*h*w (source: keys) x h*w (tar: queries)
    aff = aff.transpose(1, 0)
    tk_val, _ = torch.topk(aff, dim=0, k=15)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    # ----------------------------------------------------------------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()
    # ----------------------------------------------------------------------------------------------------------------
    # get mask
    seg_tar = torch.mm(segs, aff)
    # down sample for points of return_feat_tar
    fore_index = torch.where(seg_tar[0, :] != 0)[0]
    fore_nums = len(fore_index)
    back_index = torch.where(seg_tar[0, :] == 0)[0]
    back_nums = len(back_index)
    # generate random index
    random_indices = torch.randperm(len(fore_index))[: int(len(fore_index) * fore_nums / (fore_nums + back_nums) * sample_ratio)]
    # choice sub data from all data
    fore_index_sample = fore_index[random_indices]
    # ------------------------------------------------------------------------------------
    # generate random index
    random_indices = torch.randperm(len(back_index))[: int(len(back_index) * back_nums / (fore_nums + back_nums) * sample_ratio)]
    # choice sub data from all data
    back_index_sample = back_index[random_indices]
    # concat
    all_index = torch.cat([fore_index_sample, back_index_sample])
    # get sub data
    seg_sample = seg_tar[:, all_index]
    seg_tar = seg_tar.reshape(1, C, h, w)
    return_feat_tar = return_feat_tar[:, all_index]

    return seg_tar, seg_sample, return_feat_tar

def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)