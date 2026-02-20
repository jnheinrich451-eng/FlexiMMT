import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional, List
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import T5EncoderModel, T5Tokenizer

class RefNetLoRAProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(
            self, 
            stage=None,
            # motioninversion
            num_motion_tokens=226,
            # visualize
            attn_map_save_path=None,
            cur_layer=0,
            cur_step=0,
            save_step=[0, 10, 20, 30, 40],
            save_layer=[0, 1, 2, 3, 4],
            # use_mask
            use_mask=True
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.stage = stage
        self.attention_mask = None

        # Visualization
        self.attn_map_save_path = attn_map_save_path
        self.cur_layer = cur_layer
        self.cur_step = cur_step
        self.save_step = save_step
        self.save_layer = save_layer

        # use_mask
        self.use_mask = use_mask

        # Stage 1/2
        if stage is not None:
            self.num_motion_tokens = num_motion_tokens
            self.motion_inversion_tokens = nn.Parameter(torch.zeros(1, num_motion_tokens, 3072))
            nn.init.kaiming_uniform_(self.motion_inversion_tokens)
            # nn.init.zeros_(self.motion_inversion_tokens)

    def _get_and_update_status(self):
        """Get and update current status counter"""
        # calculate cur_step and cur_layer
        os.makedirs(self.attn_map_save_path, exist_ok=True)
        if not os.path.exists(os.path.join(self.attn_map_save_path, "status.txt")):
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write(f"0")
                cur_status = 0
        else:
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "r") as f:
                cur_status = int(f.read()) + 1
            # overwrite
            with open(os.path.join(self.attn_map_save_path, "status.txt"), "w") as f:
                f.write(f"{cur_status}")
            
        return cur_status

    def visualize_object2video_map(self, object2video_map, width=45, height=30, save_path=None):
        """
        Visualize object2video_map per frame

        Args:
            object2video_map: [frames*width*height] tensor
            width: frame width
            height: frame height
            save_path: save path
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Calculate number of frames
        frames = object2video_map.shape[-0]

        # Reshape to [frames, height, width]
        reshaped_map = object2video_map.view(frames, height, width)

        # Convert to numpy array
        np_maps = reshaped_map.detach().float().cpu().numpy()

        # Calculate global color range for consistency
        vmin, vmax = np.min(np_maps), np.max(np_maps)

        # Calculate grid layout
        num_cols = min(4, frames)  # max 4 columns
        num_rows = (frames + num_cols - 1) // num_cols

        # Create figure
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows))

        # Ensure axes is 2D array if only one row or column
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each frame
        for frame_idx in range(frames):
            row = frame_idx // num_cols
            col = frame_idx % num_cols

            ax = axes[row, col] if frames > 1 else axes

            im = ax.imshow(np_maps[frame_idx], cmap='jet', vmin=vmin, vmax=vmax)
            ax.set_title(f'Frame {frame_idx}')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Attention Weight')

        # Hide extra subplots
        for frame_idx in range(frames, num_rows * num_cols):
            row = frame_idx // num_cols
            col = frame_idx % num_cols
            if frames > 1:
                axes[row, col].set_visible(False)

        plt.suptitle('Object to Video Attention Map by Frame', fontsize=16)
        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = "object2video_visualization"
        
        os.makedirs(save_path, exist_ok=True)
        
        # Get current status for file naming
        if hasattr(self, '_get_and_update_status'):
            cur_status = self._get_and_update_status()
            num_layer = 5
            cur_step = cur_status // num_layer
            cur_layer = cur_status % num_layer
            filename = f"object2video_map_step{cur_step}_layer{cur_layer}.png"
        else:
            filename = "object2video_map.png"
        
        save_file_path = os.path.join(save_path, filename)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Object2Video visualization saved to: {save_file_path}")

    def visualize_mask(self, mask, width=45, height=30, save_path=None):
        """
        Visualize mask per frame

        Args:
            mask: [frames, height, width] boolean tensor
            width: frame width
            height: frame height
            save_path: save path
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Get number of frames
        frames = mask.shape[0]

        # Convert to numpy array
        np_masks = mask.detach().cpu().numpy()

        # Calculate grid layout
        num_cols = min(4, frames)  # max 4 columns
        num_rows = (frames + num_cols - 1) // num_cols

        # Create figure
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows))

        # Ensure axes is 2D array if only one row or column
        if num_rows == 1 and num_cols > 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1 and num_rows > 1:
            axes = axes.reshape(-1, 1)
        elif num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])

        # Plot each frame
        for frame_idx in range(frames):
            row = frame_idx // num_cols
            col = frame_idx % num_cols

            ax = axes[row, col]

            im = ax.imshow(np_masks[frame_idx], cmap='binary', vmin=0, vmax=1)
            ax.set_title(f'Frame {frame_idx}')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Mask (Top 50%)')

        # Hide extra subplots
        for frame_idx in range(frames, num_rows * num_cols):
            row = frame_idx // num_cols
            col = frame_idx % num_cols
            axes[row, col].set_visible(False)

        plt.suptitle('Object to Video Mask (Top 50%) by Frame', fontsize=16)
        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = "object2video_visualization"
        
        os.makedirs(save_path, exist_ok=True)
        
        # Get current status for file naming
        if hasattr(self, '_get_and_update_status'):
            cur_status = self._get_and_update_status()
            num_layer = 5
            cur_step = cur_status // num_layer
            cur_layer = cur_status % num_layer
            filename = f"object2video_mask_step{cur_step}_layer{cur_layer}.png"
        else:
            filename = "object2video_mask.png"
        
        save_file_path = os.path.join(save_path, filename)
        plt.savefig(save_file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Object2Video mask visualization saved to: {save_file_path}")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        motion_indices: List = None
    ) -> torch.Tensor:
        concept_pos = motion_indices[0]
        motion_pos = motion_indices[1]

        text_seq_length = encoder_hidden_states.size(1)
        if self.stage is not None:
            # FAE
            cat_tokens = self.motion_inversion_tokens.repeat(encoder_hidden_states.size(0), 1, 1)
            encoder_hidden_states = torch.cat([encoder_hidden_states, cat_tokens], dim=1)
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        else:
            # RefAdapter
            encoder_h_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # Regular QKV calculation (potentially post-FAE)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, encoder_h_seq_length:] = apply_rotary_emb(query[:, :, encoder_h_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, encoder_h_seq_length:] = apply_rotary_emb(key[:, :, encoder_h_seq_length:], image_rotary_emb)


        if self.use_mask:
            # Get mask of object's corresponding video_embedding from text_embedding
            # A cat turning its head
            query_object = query[0,:,concept_pos[0]:concept_pos[0]+concept_pos[1],:]
            key_video = key[0,:,text_seq_length+self.num_motion_tokens:]
            # print((query_object @ key_video.transpose(-2, -1)).shape)
            object2video_map = torch.mean(query_object @ key_video.transpose(-2, -1),dim=(0,1)) # [1,frames*width*height]
            width, height = 45, 30
            frames = int(object2video_map.shape[-1] / (width*height))

            # Reshape to [frames, height, width] for per-frame processing
            object2video_map = object2video_map.reshape(frames, height, width)

            # Create mask for each frame
            mask = torch.zeros_like(object2video_map, dtype=torch.bool)
            for f in range(frames):
                frame_values = object2video_map[f].flatten().float()
                # Calculate mean as threshold
                threshold = torch.mean(frame_values)
                # Create mask: values above mean are True (not masked), below mean are False (masked)
                frame_mask = object2video_map[f] > threshold
                mask[f] = frame_mask
            
            mask_flattened = mask.reshape(-1)
            
            # Randomly select 30% of unmasked regions to mask
            # Get indices of unmasked positions
            unmasked_indices = torch.where(mask_flattened)[0]
            if len(unmasked_indices) > 0:
                # Randomly select 30% of unmasked indices
                num_to_mask = int(len(unmasked_indices) * 0.3)
                # Ensure at least one position is selected
                num_to_mask = max(1, num_to_mask)
                # Randomly select specified number of indices
                indices_to_mask = unmasked_indices[torch.randperm(len(unmasked_indices))[:num_to_mask]]
                # Create new mask, initialized to all True
                to_mask_indices = torch.ones_like(mask_flattened)
                # Set positions to be masked to False
                to_mask_indices[indices_to_mask] = False
                # Update mask: keep original unmasked, but remove randomly selected 30%
                mask_flattened = mask_flattened & to_mask_indices
            else:
                mask_flattened = mask_flattened  # unchanged
            
            # self.visualize_mask(
            #     mask,
            #     width=width,
            #     height=height,
            #     save_path=self.attn_map_save_path
            # )
            
            # self.visualize_object2video_map(
            #     object2video_map, 
            #     width=width, 
            #     height=height, 
            #     # save_path=None,
            #     save_path=self.attn_map_save_path
            # )

            if self.stage is not None:
                q_len = query.shape[-2]
                # Mask out parts below mean
                attention_mask = torch.zeros((q_len, q_len), device=query.device, dtype=query.dtype)
                attention_mask[:,text_seq_length:text_seq_length+self.num_motion_tokens] = float("-inf")
                attention_mask[text_seq_length:text_seq_length+self.num_motion_tokens, :] = float("-inf")

                # Set video parts above mean to 0
                video_start_idx = text_seq_length + self.num_motion_tokens
                # Find positions above mean
                greater_indices = torch.where(mask_flattened)[0] + video_start_idx
                # Set above-mean parts to 0
                attention_mask[text_seq_length:text_seq_length+self.num_motion_tokens, greater_indices] = 0
                attention_mask[greater_indices, text_seq_length:text_seq_length+self.num_motion_tokens] = 0

                # Attend to motion words
                attention_mask[motion_pos[0]:motion_pos[0]+motion_pos[1],text_seq_length:text_seq_length+self.num_motion_tokens] = 0
                attention_mask[text_seq_length:text_seq_length+self.num_motion_tokens, motion_pos[0]:motion_pos[0]+motion_pos[1]] = 0

                # Enable m2m attention
                attention_mask[text_seq_length:text_seq_length+self.num_motion_tokens,text_seq_length:text_seq_length+self.num_motion_tokens] = 0

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [encoder_h_seq_length, hidden_states.size(1) - encoder_h_seq_length], dim=1
        )
        encoder_hidden_states = encoder_hidden_states[:, :text_seq_length, :]
        return hidden_states, encoder_hidden_states