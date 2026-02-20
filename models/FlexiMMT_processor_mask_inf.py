import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional, List
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import T5EncoderModel, T5Tokenizer
import einops
import math
    
class RefNetLoRAProcessor(nn.Module):
    def __init__(
            self, 
            stage=None,
            num_motion_tokens=226,
            use_tokens=None,  # List indicating which tokens to use
            first_attns=None,  # List storing attention range for each token
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.stage = stage
        
        # Initialize token usage flags and attention ranges
        self.use_tokens = use_tokens if use_tokens is not None else []
        self.first_attns = first_attns if first_attns is not None else []
        
        # Ensure both lists have same length
        max_len = max(len(self.use_tokens), len(self.first_attns))
        self.use_tokens = self.use_tokens + [False] * (max_len - len(self.use_tokens))
        self.first_attns = self.first_attns + [None] * (max_len - len(self.first_attns))
        
        # Record total number of tokens
        self.num_tokens = len(self.use_tokens)
        
        # Stage 1/2
        if stage is not None:
            self.num_motion_tokens = num_motion_tokens
            
            # Create parameters for each token to use
            for i, use_token in enumerate(self.use_tokens):
                if use_token:
                    # Dynamically create token parameters
                    setattr(self, f"tokens_{i}", nn.Parameter(torch.zeros(1, num_motion_tokens, 3072)))
                    # Initialize
                    nn.init.kaiming_uniform_(getattr(self, f"tokens_{i}"))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_masks=None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: int = 0,
        mask_computation_stopped: bool = False,
        purn_time: int = 999
    ):
        text_seq_length = encoder_hidden_states.size(1)
        
        # Prepare hidden_states with motion tokens for final computation
        if self.stage is not None:
            # Count number of used tokens
            num_used_tokens = sum(self.use_tokens)
            
            if num_used_tokens > 0:
                # Collect all used tokens
                all_tokens = []
                for i, use_token in enumerate(self.use_tokens):
                    if use_token:
                        token = getattr(self, f"tokens_{i}").repeat(encoder_hidden_states.size(0), 1, 1)
                        all_tokens.append(token)
                
                # Concatenate tokens if any
                if all_tokens:
                    cat_tokens = torch.cat(all_tokens, dim=1)
                    # Concatenate to encoder_hidden_states
                    encoder_hidden_states = torch.cat([encoder_hidden_states, cat_tokens], dim=1)
            
            encoder_h_seq_length = encoder_hidden_states.size(1)
            final_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        else:
            # RefAdapter
            encoder_h_seq_length = encoder_hidden_states.size(1)
            final_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # Step 2: Compute final result using hidden_states with motion tokens
        query_final = attn.to_q(final_hidden_states)
        key_final = attn.to_k(final_hidden_states)
        value_final = attn.to_v(final_hidden_states)

        inner_dim = key_final.shape[-1]
        head_dim = inner_dim // attn.heads

        query_final = query_final.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_final = key_final.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_final = value_final.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query_final = attn.norm_q(query_final)
        if attn.norm_k is not None:
            key_final = attn.norm_k(key_final)

        # Apply RoPE if needed (for final)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query_final[:, :, encoder_h_seq_length:] = apply_rotary_emb(query_final[:, :, encoder_h_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key_final[:, :, encoder_h_seq_length:] = apply_rotary_emb(key_final[:, :, encoder_h_seq_length:], image_rotary_emb)
        
        filter_token = False
        if filter_token and mask_computation_stopped:
            # Pre-allocate lists
            num_tokens = len(self.use_tokens)
            filtered_indices_list = [None] * num_tokens
            filtered_token_nums = [0] * num_tokens
            
            token_offset = 0
            for i, use_token in enumerate(self.use_tokens):
                if use_token:
                    token_start = text_seq_length + token_offset
                    token_end = token_start + self.num_motion_tokens
                    
                    # Compute full attention (cannot be segmented)
                    motion_map_full = torch.softmax(
                        query_final[1:, :, token_start:token_end] @ key_final[1:].transpose(-2, -1),
                        dim=-1
                    )

                    # Optimization 1: extract only needed parts, immediately delete full map
                    video_start_in_full = text_seq_length + sum(self.use_tokens) * self.num_motion_tokens
                    motion_map_video = motion_map_full[:, :, :, video_start_in_full:].clone()
                    motion_map_text = motion_map_full[:, :, :, :text_seq_length].clone()
                    del motion_map_full  # Immediately release large tensor
                    
                    motion_map = torch.cat([motion_map_video, motion_map_text], dim=-1)
                    del motion_map_video, motion_map_text

                    # Modify token filtering logic near line 149
                    motion_to_all_ratio = torch.mean(torch.sum(motion_map, dim=-1), dim=1)[0]
                    del motion_map

                    # Dynamically compute number of tokens to keep based on timestep
                    # timestep: 999 -> 0, keep count: num_motion_tokens -> 1
                    max_tokens = self.num_motion_tokens  # 226
                    min_tokens = 1
                    # # Option 1: cosine decay (recommended)
                    # # Slow change early, accelerated decay later
                    # normalized_t = timestep[0] / 999.0  # Normalize to [0, 1]
                    # # Segmented rapid decay strategy
                    # gamma=0.9
                    # beta=2
                    # if normalized_t > gamma:
                    #     # First half: stay stable, slow decay
                    #     keep_ratio = 0.5 * (1 + math.cos(math.pi * (1 - normalized_t)))
                    # else:
                    #     # Second half: rapid decay
                    #     adjusted_t = normalized_t / gamma  # Map to [0,1]
                    #     keep_ratio = adjusted_t ** 3  # Cubic decay, very steep

                    normalized_t = timestep[0] / purn_time[0]
                    # print(normalized_t)
                    adjusted_t = normalized_t  # Map to [0,1]
                    keep_ratio = adjusted_t ** 4 # 4th power decay

                    num_keep_tokens = int(min_tokens + (max_tokens - min_tokens) * keep_ratio)
                    num_keep_tokens = max(min_tokens, min(num_keep_tokens, max_tokens))  # Ensure within [1, 226] range
                    # print(keep_ratio, num_keep_tokens)

                    # Use topk to select most important tokens, count determined by timestep
                    _, sorted_indices = torch.topk(motion_to_all_ratio, k=self.num_motion_tokens, largest=True)
                    # Only keep top num_keep_tokens most important tokens
                    filtered_idx = sorted_indices[:num_keep_tokens]
                    del sorted_indices, motion_to_all_ratio

                    filtered_indices_list[i] = filtered_idx
                    filtered_token_nums[i] = filtered_idx.shape[0]
                    token_offset += self.num_motion_tokens

            total_filtered_tokens = sum(filtered_token_nums)
            encoder_h_seq_length = text_seq_length + total_filtered_tokens
            
            if total_filtered_tokens > 0:
                # Optimization 4: collect three tensor parts separately, avoid tuple lists
                parts_q = [query_final[:, :, :text_seq_length]]
                parts_k = [key_final[:, :, :text_seq_length]]
                parts_v = [value_final[:, :, :text_seq_length]]
                
                token_offset = 0
                for i, use_token in enumerate(self.use_tokens):
                    if use_token and filtered_indices_list[i] is not None:
                        token_start = text_seq_length + token_offset
                        filtered_idx = filtered_indices_list[i] + token_start
                        
                        parts_q.append(query_final[:, :, filtered_idx])
                        parts_k.append(key_final[:, :, filtered_idx])
                        parts_v.append(value_final[:, :, filtered_idx])
                        
                        token_offset += self.num_motion_tokens
                
                video_start = text_seq_length + sum(self.use_tokens) * self.num_motion_tokens
                parts_q.append(query_final[:, :, video_start:])
                parts_k.append(key_final[:, :, video_start:])
                parts_v.append(value_final[:, :, video_start:])
                
                query_final = torch.cat(parts_q, dim=2)
                key_final = torch.cat(parts_k, dim=2)
                value_final = torch.cat(parts_v, dim=2)
                del parts_q, parts_k, parts_v

                parts_hidden = [encoder_hidden_states[:, :text_seq_length]]
                
                token_offset = 0
                for i, use_token in enumerate(self.use_tokens):
                    if use_token and filtered_indices_list[i] is not None:
                        token_start = text_seq_length + token_offset
                        filtered_idx = filtered_indices_list[i]
                        parts_hidden.append(encoder_hidden_states[:, token_start + filtered_idx])
                        token_offset += self.num_motion_tokens
                
                video_start = text_seq_length + sum(self.use_tokens) * self.num_motion_tokens
                parts_hidden.append(final_hidden_states[:, video_start:])
                
                final_hidden_states = torch.cat(parts_hidden, dim=1)
                del parts_hidden
        
        # Build final attention mask (version with motion tokens)
        q_len = query_final.shape[-2]
        final_attention_mask = torch.zeros((q_len, q_len), device=query_final.device, dtype=query_final.dtype)

        if self.stage is not None:
            # Calculate video start index - use filtered length
            if filter_token and mask_computation_stopped and total_filtered_tokens > 0:
                video_start_idx = encoder_h_seq_length  # Already includes filtered tokens
            else:
                video_start_idx = text_seq_length
                if sum(self.use_tokens) > 0:
                    video_start_idx += self.num_motion_tokens * sum(self.use_tokens)
            
            # Process all used tokens and masks
            greater_indices = []
            
            if concept_masks is not None and len(concept_masks) >= len(self.use_tokens):
                # Prepare mask indices for each token
                for i, use_token in enumerate(self.use_tokens):
                    if use_token and i < len(concept_masks):
                        mask_flattened = concept_masks[i]
                        indices = torch.where(mask_flattened)[0] + video_start_idx
                        greater_indices.append(indices)
                    else:
                        greater_indices.append(None)
            
            # Handle attention between tokens in text
            # Prevent tokens from attending to each other (set for all pairs)
            for i in range(len(self.first_attns)):
                if self.first_attns[i] is not None:
                    for j in range(i+1, len(self.first_attns)):
                        if self.first_attns[j] is not None:
                            start_i, end_i = self.first_attns[i]
                            start_j, end_j = self.first_attns[j]
                            final_attention_mask[start_i:end_i, start_j:end_j] = float("-inf")
                            final_attention_mask[start_j:end_j, start_i:end_i] = float("-inf")
            
            # Process attention for each token
            token_offset = 0
            for i, use_token in enumerate(self.use_tokens):
                if use_token and self.first_attns[i] is not None:
                    start_idx, end_idx = self.first_attns[i]
                    token_start = text_seq_length + token_offset
                    
                    # Use filtered token count
                    if filter_token and mask_computation_stopped and filtered_token_nums[i] > 0:
                        token_length = filtered_token_nums[i]
                    else:
                        token_length = self.num_motion_tokens
                    
                    token_end = token_start + token_length
                    
                    # Text motion part attends to specified video parts
                    if len(greater_indices) != 0 and greater_indices[i] is not None:
                        # By default token does not attend to video
                        final_attention_mask[start_idx:end_idx, video_start_idx:] = float("-inf")
                        final_attention_mask[video_start_idx:, start_idx:end_idx] = float("-inf")
                        # Only attend to specific regions
                        final_attention_mask[start_idx:end_idx, greater_indices[i]] = 0
                        final_attention_mask[greater_indices[i], start_idx:end_idx] = 0
                    
                    # Set token mask
                    # By default all tokens are not attended to
                    final_attention_mask[:, token_start:token_end] = float("-inf")
                    final_attention_mask[token_start:token_end, :] = float("-inf")

                    # # Enable m2m attention
                    # final_attention_mask[token_start:token_end, token_start:token_end] = 0
                    
                    # Motion token attends to specified video parts
                    if len(greater_indices) != 0 and greater_indices[i] is not None:
                        final_attention_mask[token_start:token_end, greater_indices[i]] = 0
                        final_attention_mask[greater_indices[i], token_start:token_end] = 0
                    
                    # Motion token attends to corresponding text part
                    final_attention_mask[start_idx:end_idx, token_start:token_end] = 0
                    final_attention_mask[token_start:token_end, start_idx:end_idx] = 0
                    
                    token_offset += token_length  # Accumulate using actual token length
        
        # Compute final attention
        hidden_states_new = F.scaled_dot_product_attention(
            query_final, key_final, value_final, attn_mask=final_attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_new = hidden_states_new.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        
        # linear proj
        hidden_states_new = attn.to_out[0](hidden_states_new)
        # dropout
        hidden_states_new = attn.to_out[1](hidden_states_new)

        encoder_hidden_states, hidden_states_new = hidden_states_new.split(
            [encoder_h_seq_length, hidden_states_new.size(1) - encoder_h_seq_length], dim=1
        )
        encoder_hidden_states = encoder_hidden_states[:, :text_seq_length, :]

        return hidden_states_new, encoder_hidden_states