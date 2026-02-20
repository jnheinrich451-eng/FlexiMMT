import pandas as pd
import numpy as np
import json

from PIL import Image
import random


import argparse
import logging
from decord import VideoReader, cpu, gpu
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.functional as F
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT


def resize_wo_crop(arr, image_size):
    arr = resize(
        arr,
        size=[image_size[0], image_size[1]],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return arr


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    '''
    Resize the array to the given image size.
    Args:
        arr: np.ndarray, shape: [num_frames, channels, height, width]
        image_size: tuple, (height, width)
        reshape_mode: str, "random" or "center"
    Returns:
        np.ndarray, shape: [num_frames, channels, height, width]
    '''

    # print(f"arr shape: {arr.shape}; image_size: {image_size}")
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
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
    arr = transforms.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    if len(arr.shape) == 3:
        arr = arr.unsqueeze(0)
    return arr

class VideoInpaintingDataset(Dataset):
    def __init__(
        self,
        meta_file_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        id_token: Optional[str] = None,
        is_train: bool = True,
        load_mode: str = "online",
    ) -> None:
        super().__init__()
        self.meta_file_path = Path(meta_file_path)
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.id_token = id_token or ""
        self.is_train = is_train
        self.load_mode = load_mode
        # init_time = time.time()
        self.instance_metas = self._load_dataset_from_local_path()
        self.instance_metas = self._preprocess_data_online()
        # logger.info(f"Dataset loaded in {timedelta(seconds=int(time.time() - init_time))}")

    def __len__(self):
        return len(self.instance_metas)

    def __getitem__(self, index):
        # load_video_time = time.time()
        while True:
            video_path, start_frame, end_frame, fps, mask_id, prompt, concept_words, motion_words = self.instance_metas[index]
            
            cpu_idx = random.randint(0, os.cpu_count() - 1)
            vr = VideoReader(video_path, ctx=cpu(cpu_idx))
            frames = vr.get_batch(range(start_frame, end_frame)).asnumpy()
            video = np.array(frames)
            # Align to the 8 fps
            video = video[::int(fps//8)]
            if video.shape[0] < self.max_num_frames:
                index += 1
                continue
            # print(f"#### Video Load Time: {time.time() - load_video_time}s ####")
            if type(prompt) != str:
                prompt = ""
            return {
                "prompt": self.id_token + prompt,
                "video": video,
                "concept_words": concept_words,
                "motion_words": motion_words
            }

    def _load_dataset_from_local_path(self):
        '''
        read the meta file and corrupt file, and return the metas
        '''
        if not self.meta_file_path.exists():
            raise ValueError("Meta file does not exist")
        
        metas = pd.read_csv(self.meta_file_path)
        if self.is_train:
            metas = metas[:]
            # metas = metas[metas['caption'].str.len() > 50]
            metas = metas.values
        else:
            metas = metas[:]
            metas = metas.values
        return metas

    def _preprocess_data_online(self):
        return self.instance_metas

class MyWebDataset():
    def __init__(self,resolution, tokenizer, max_num_frames, max_sequence_length, proportion_empty_prompts, is_train=True, emptytxt=False, use_different_first_frame=False):
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames
        self.val_max_num_frames = max_num_frames
        self.max_sequence_length = max_sequence_length
        self.proportion_empty_prompts = proportion_empty_prompts
        self.is_train = is_train
        self.emptytxt = emptytxt
        self.use_different_first_frame = use_different_first_frame

    def tokenize_captions(self, caption, is_train=True):
        # uncond_prob = self.proportion_empty_prompts
        uncond_prob = 0.1
        prob = random.random()
        if (prob < uncond_prob) and is_train:
        # if random.random() < self.proportion_empty_prompts:
            caption=""
        elif isinstance(caption, str):
            caption=caption
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption=random.choice(caption) if is_train else caption[0]
        else:
            raise ValueError(
                f"Caption column should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            caption, max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def call_single_example(self, example, pixel_values, input_ids, image_values, captions):
        caption=example["prompt"] if not self.emptytxt else "best quality, high quality"
        video=example["video"]
        video_ori = video
        frame, height, width, c = video.shape

        if frame > self.max_num_frames:
            begin_idx = random.randint(0, frame - self.max_num_frames)
            end_idx = begin_idx + self.max_num_frames
            video = video[begin_idx:end_idx]
            frame = end_idx - begin_idx
        elif frame <= self.max_num_frames:
            remainder = (3 + (frame % 4)) % 4
            if remainder != 0:
                video = video[:-remainder]
            frame = video.shape[0]
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        if self.use_different_first_frame:
            # Randomly select frames from ori_video
            idx_list = random.sample(range(video_ori.shape[0]), frame)
            video_ori = video_ori[idx_list]
            image = torch.from_numpy(video_ori).permute(0, 3, 1, 2)
        else:
            image = video

        video = resize_wo_crop(video, self.resolution)
        image = resize_wo_crop(image, self.resolution)
        # video = resize_for_rectangle_crop(video, self.resolution, reshape_mode="center")
        # image = resize_for_rectangle_crop(image, self.resolution, reshape_mode="center")
        video = video.permute(0, 2, 3, 1).numpy()
        image = image.permute(0, 2, 3, 1).numpy()
        video = (video.astype(np.float32) / 127.5) - 1.0
        image = (image.astype(np.float32) / 127.5) - 1.0
        if video.max() > 1.0 or video.min() < -1.0:
            print(video.max(), video.min())
            raise ValueError("Image pixel values should be in the range [-1, 1]")

        pixel_values.append(torch.tensor(video).permute(0, 3, 1, 2).unsqueeze(0))
        image_values.append(torch.tensor(image).permute(0, 3, 1, 2).unsqueeze(0))
        if not self.is_train:
            input_ids.append(self.tokenize_captions("")[0].unsqueeze(0))
        input_ids.append(self.tokenize_captions(caption)[0].unsqueeze(0))
        captions.append(caption)

        return pixel_values, input_ids, image_values, captions

    def get_motion_token_indices(self, tokenizer, prompt, motion_words):
        """
        Automatically determine token index positions and lengths of motion words in the prompt text

        Args:
            tokenizer: CogVideoX model tokenizer
            prompt: Text prompt for video generation
            motion_words: List of motion words, e.g. ["rotate", "turning"]

        Returns:
            List of (index, length) tuples for each motion word
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

    def __call__(self, examples):
        pixel_values=[]
        image_values=[]
        input_ids=[]
        captions=[]
        motion_indices_list=[]
        
        max_num_frames = self.max_num_frames if self.is_train else self.val_max_num_frames
        for example in examples:
            caption=example["prompt"] if not self.emptytxt else ""
            video=example["video"] # frame, height, width, c

            concept_word = example["concept_words"]
            motion_word = example["motion_words"]

            motion_indices = self.get_motion_token_indices(self.tokenizer, caption, [concept_word, motion_word])
            motion_indices_list.append(motion_indices)

            video_ori = video
            frame, height, width, c = video.shape
            
            if frame > max_num_frames:
                begin_idx = random.randint(0, frame - max_num_frames)
                end_idx = begin_idx + max_num_frames
                video = video[begin_idx:end_idx]
                frame = end_idx - begin_idx
            elif frame <= max_num_frames:
                # TODO: (8*k+1)
                remainder = (3 + (frame % 4)) % 4
                if remainder != 0:
                    video = video[:-remainder]
                frame = video.shape[0]

            video = torch.from_numpy(video).permute(0, 3, 1, 2) # [F, C, H, W]
            if self.use_different_first_frame:
                # Randomly select frames from ori_video
                idx_list = random.sample(range(video_ori.shape[0]), frame)
                video_ori = video_ori[idx_list]
                image = torch.from_numpy(video_ori).permute(0, 3, 1, 2)
            else:
                image = video # [1, C, H, W]
            
            video = resize_wo_crop(video, self.resolution)
            image = resize_wo_crop(image, self.resolution)
            # video = resize_for_rectangle_crop(video, self.resolution, reshape_mode="center")
            # image = resize_for_rectangle_crop(image, self.resolution, reshape_mode="center")
            video = video.permute(0, 2, 3, 1).numpy()
            image = image.permute(0, 2, 3, 1).numpy()
            video = (video.astype(np.float32) / 127.5) - 1.0
            image = (image.astype(np.float32) / 127.5) - 1.0
            if video.max() > 1.0 or video.min() < -1.0:
                print(video.max(), video.min())
                raise ValueError("Image pixel values should be in the range [-1, 1]")

            pixel_values.append(torch.tensor(video).permute(0, 3, 1, 2))
            image_values.append(torch.tensor(image).permute(0, 3, 1, 2))
            if not self.is_train:
                input_ids.append(self.tokenize_captions("")[0])
            input_ids.append(self.tokenize_captions(caption)[0])
            captions.append(caption)


        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        image_values = torch.stack(image_values)
        image_values = image_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack(input_ids)


        return {
            "pixel_values": pixel_values, # B, T, C, H, W
            "image_values": image_values, # B, 1, C, H, W
            "input_ids": input_ids, # B, L
            "captions": captions,
            "motion_indices_list": motion_indices_list
        }


class VideoInpaintingDataset2(Dataset):
    def __init__(
        self,
        meta_file_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        id_token: Optional[str] = None,
        is_train: bool = True,
        load_mode: str = "online",
    ) -> None:
        super().__init__()
        self.meta_file_path = Path(meta_file_path)
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.id_token = id_token or ""
        self.is_train = is_train
        self.load_mode = load_mode
        # init_time = time.time()
        self.instance_metas = self._load_dataset_from_local_path()
        self.instance_metas = self._preprocess_data_online()
        # logger.info(f"Dataset loaded in {timedelta(seconds=int(time.time() - init_time))}")

    def __len__(self):
        return len(self.instance_metas)

    def __getitem__(self, index):
        # load_video_time = time.time()
        while True:
            video_path, start_frame, end_frame, fps, mask_id, prompt, concept_words, motion_words = self.instance_metas[index]
            
            mask_path = video_path.replace('reference_videos', 'reference_video_masks_train').split(".")[0]
            mask_files = sorted([
                f for f in os.listdir(mask_path) if f.endswith('.png') and f.split('.')[0].isdigit()
            ], key=lambda x: int(x.split('.')[0]))

            mask = np.stack([
                np.array(Image.open(os.path.join(mask_path, f)).convert('L')) for f in mask_files
            ])

            cpu_idx = random.randint(0, os.cpu_count() - 1)
            vr = VideoReader(video_path, ctx=cpu(cpu_idx))
            frames = vr.get_batch(range(start_frame, end_frame)).asnumpy()
            video = np.array(frames)
            # Align to the 8 fps
            video = video[::int(fps//8)]
            if video.shape[0] < self.max_num_frames:
                index += 1
                continue
            # print(f"#### Video Load Time: {time.time() - load_video_time}s ####")
            if type(prompt) != str:
                prompt = ""
            return {
                "prompt": self.id_token + prompt,
                "video": video,
                "mask": mask,
                "concept_words": concept_words,
                "motion_words": motion_words
            }

    def _load_dataset_from_local_path(self):
        '''
        read the meta file and corrupt file, and return the metas
        '''
        if not self.meta_file_path.exists():
            raise ValueError("Meta file does not exist")
        
        metas = pd.read_csv(self.meta_file_path)
        # if self.is_train:
        #     metas = metas[:]
        #     # metas = metas[metas['caption'].str.len() > 50]
        #     metas = metas.values
        # else:
        #     metas = metas[:]
        #     metas = metas.values
        metas = metas[:]
        metas = metas.values
        
        return metas

    def _preprocess_data_online(self):
        return self.instance_metas


class MyWebDataset2():
    def __init__(self,resolution, tokenizer, max_num_frames, max_sequence_length, proportion_empty_prompts, is_train=True, emptytxt=False, use_different_first_frame=False):
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames
        self.val_max_num_frames = max_num_frames
        self.max_sequence_length = max_sequence_length
        self.proportion_empty_prompts = proportion_empty_prompts
        self.is_train = is_train
        self.emptytxt = emptytxt
        self.use_different_first_frame = use_different_first_frame

    def tokenize_captions(self, caption, is_train=True):
        # uncond_prob = self.proportion_empty_prompts
        uncond_prob = 0.1
        prob = random.random()
        if (prob < uncond_prob) and is_train:
        # if random.random() < self.proportion_empty_prompts:
            caption=""
        elif isinstance(caption, str):
            caption=caption
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption=random.choice(caption) if is_train else caption[0]
        else:
            raise ValueError(
                f"Caption column should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            caption, max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def call_single_example(self, example, pixel_values, input_ids, image_values, captions):
        caption=example["prompt"] if not self.emptytxt else "best quality, high quality"
        video=example["video"]
        video_ori = video
        frame, height, width, c = video.shape

        if frame > self.max_num_frames:
            begin_idx = random.randint(0, frame - self.max_num_frames)
            end_idx = begin_idx + self.max_num_frames
            video = video[begin_idx:end_idx]
            frame = end_idx - begin_idx
        elif frame <= self.max_num_frames:
            remainder = (3 + (frame % 4)) % 4
            if remainder != 0:
                video = video[:-remainder]
            frame = video.shape[0]
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        if self.use_different_first_frame:
            # Randomly select frames from ori_video
            idx_list = random.sample(range(video_ori.shape[0]), frame)
            video_ori = video_ori[idx_list]
            image = torch.from_numpy(video_ori).permute(0, 3, 1, 2)
        else:
            image = video

        video = resize_wo_crop(video, self.resolution)
        image = resize_wo_crop(image, self.resolution)
        # video = resize_for_rectangle_crop(video, self.resolution, reshape_mode="center")
        # image = resize_for_rectangle_crop(image, self.resolution, reshape_mode="center")
        video = video.permute(0, 2, 3, 1).numpy()
        image = image.permute(0, 2, 3, 1).numpy()
        video = (video.astype(np.float32) / 127.5) - 1.0
        image = (image.astype(np.float32) / 127.5) - 1.0
        if video.max() > 1.0 or video.min() < -1.0:
            print(video.max(), video.min())
            raise ValueError("Image pixel values should be in the range [-1, 1]")

        pixel_values.append(torch.tensor(video).permute(0, 3, 1, 2).unsqueeze(0))
        image_values.append(torch.tensor(image).permute(0, 3, 1, 2).unsqueeze(0))
        if not self.is_train:
            input_ids.append(self.tokenize_captions("")[0].unsqueeze(0))
        input_ids.append(self.tokenize_captions(caption)[0].unsqueeze(0))
        captions.append(caption)

        return pixel_values, input_ids, image_values, captions

    def get_motion_token_indices(self, tokenizer, prompt, motion_words):
        """
        Automatically determine token index positions and lengths of motion words in the prompt text

        Args:
            tokenizer: CogVideoX model tokenizer
            prompt: Text prompt for video generation
            motion_words: List of motion words, e.g. ["rotate", "turning"]

        Returns:
            List of (index, length) tuples for each motion word
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

    def __call__(self, examples):
        pixel_values=[]
        mask_values=[]
        image_values=[]
        input_ids=[]
        captions=[]
        motion_indices_list=[]
        
        max_num_frames = self.max_num_frames if self.is_train else self.val_max_num_frames
        for example in examples:
            caption = example["prompt"] if not self.emptytxt else ""
            video = example["video"]      # (frame, height, width, c)
            mask = example["mask"]        # (frame, height, width)

            concept_word = example["concept_words"]
            motion_word = example["motion_words"]

            motion_indices = self.get_motion_token_indices(self.tokenizer, caption, [concept_word, motion_word])
            motion_indices_list.append(motion_indices)

            video_ori = video
            frame, height, width, c = video.shape
            
            if frame > max_num_frames:
                begin_idx = random.randint(0, frame - max_num_frames)
                end_idx = begin_idx + max_num_frames
                video = video[begin_idx:end_idx]
                frame = end_idx - begin_idx
            elif frame <= max_num_frames:
                remainder = (3 + (frame % 4)) % 4
                if remainder != 0:
                    video = video[:-remainder]
                frame = video.shape[0]

            frame_mask = mask.shape[0]
            if frame_mask > max_num_frames:
                mask = mask[begin_idx:end_idx]
                frame_mask = end_idx - begin_idx
            elif frame_mask <= max_num_frames:
                remainder = (3 + (frame_mask % 4)) % 4
                if remainder != 0:
                    mask = mask[:-remainder]
                frame_mask = mask.shape[0]

            video = torch.from_numpy(video).permute(0, 3, 1, 2)  # [F, C, H, W]
            mask = torch.from_numpy(mask).float()                # [F, H, W]

            if self.use_different_first_frame:
                idx_list = random.sample(range(video_ori.shape[0]), frame)
                video_ori = video_ori[idx_list]
                image = torch.from_numpy(video_ori).permute(0, 3, 1, 2)
            else:
                image = video

            video = resize_wo_crop(video, self.resolution)
            # mask doesn't work with resize_wo_crop since it has no channel dim; add fake channel dim for resize
            mask = mask.unsqueeze(1)          # (F,1,H,W)
            mask = resize_wo_crop(mask, (30, 45))  # (F,1,H,W)
            mask = mask.squeeze(1)            # (F,H,W)

            image = resize_wo_crop(image, self.resolution)

            video = video.permute(0, 2, 3, 1).numpy()
            # mask is now (F,H,W), keep as float, no permute needed
            mask = mask[(0,4,8,12,16,20,24,28,32,36,40,44,48), :, :].numpy()
            image = image.permute(0, 2, 3, 1).numpy()

            video = (video.astype(np.float32) / 127.5) - 1.0
            image = (image.astype(np.float32) / 127.5) - 1.0

            if video.max() > 1.0 or video.min() < -1.0:
                print(video.max(), video.min())
                raise ValueError("Image pixel values should be in the range [-1, 1]")

            pixel_values.append(torch.tensor(video).permute(0, 3, 1, 2))  # (F,C,H,W)
            mask_values.append(torch.tensor(mask))                        # (F,H,W)
            image_values.append(torch.tensor(image).permute(0, 3, 1, 2))  # (F,C,H,W)

            if not self.is_train:
                input_ids.append(self.tokenize_captions("")[0])
            input_ids.append(self.tokenize_captions(caption)[0])
            captions.append(caption)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        mask_values = torch.stack(mask_values)  # (B, T, H, W)
        mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
        image_values = torch.stack(image_values)
        image_values = image_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack(input_ids)

        return {
            "pixel_values": pixel_values,       # B, T, C, H, W
            "mask_values": mask_values>127,         # B, T, H, W
            "image_values": image_values,       # B, T, C, H, W
            "input_ids": input_ids,             # B, L
            "captions": captions,
            "motion_indices_list": motion_indices_list
        }