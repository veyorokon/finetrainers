import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import datasets.data_files
import datasets.distributed
import datasets.exceptions
import huggingface_hub
import huggingface_hub.errors
import numpy as np
import PIL.Image
import PIL.JpegImagePlugin
import torch
import torch.distributed.checkpoint.stateful
import torchvision
from diffusers.utils import load_image, load_video
from huggingface_hub import list_repo_files, repo_exists, snapshot_download
from tqdm.auto import tqdm

from finetrainers import constants
from finetrainers import functional as FF
from finetrainers.logging import get_logger
from finetrainers.utils import find_files
from finetrainers.utils.import_utils import is_datasets_version

logger = get_logger()

MAX_PRECOMPUTABLE_ITEMS_LIMIT = 1024


def _read_caption_from_file(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read().strip()


class PatternReferenceDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    """Dataset that pairs videos/images with reference images using filename patterns.
    
    Structure:
    - video_id.mp4         # Target video
    - video_id.txt         # Text caption/prompt
    - video_id_pattern1.png # First reference image (e.g., video_id_object.png)
    - video_id_pattern2.png # Second reference image (e.g., video_id_background.png)
    """
    def __init__(
        self, 
        root: str, 
        reference_suffixes: List[str] = None,
        dataset_type: str = "video",
        infinite: bool = False
    ) -> None:
        super().__init__()
        
        self.root = pathlib.Path(root)
        self.reference_suffixes = reference_suffixes or ["_object", "_background"]
        self.dataset_type = dataset_type
        self.infinite = infinite
        
        # Find all target files based on dataset_type
        data = []
        target_files = []
        
        if dataset_type in ["video", "video_references"]:
            for ext in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
                target_files.extend(find_files(self.root.as_posix(), f"*.{ext}", depth=0))
        else:  # image
            for ext in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
                target_files.extend(find_files(self.root.as_posix(), f"*.{ext}", depth=0))
        
        for target_file in target_files:
            target_path = pathlib.Path(target_file)
            base_name = target_path.stem
            caption_path = target_path.with_suffix(".txt")
            
            if not caption_path.exists():
                logger.warning(f"Missing caption for {target_file}, skipping")
                continue
                
            # Find reference images based on patterns
            reference_images = {}
            for suffix in self.reference_suffixes:
                for ext in constants.SUPPORTED_IMAGE_FILE_EXTENSIONS:
                    ref_path = self.root / f"{base_name}{suffix}.{ext}"
                    if ref_path.exists():
                        # Extract reference type from suffix (remove leading underscore if present)
                        ref_type = suffix[1:] if suffix.startswith("_") else suffix
                        reference_images[ref_type] = ref_path.as_posix()
                        logger.info(f"Found reference image for {base_name}: {ref_type} at {ref_path}")
                        break
            
            # Skip if no reference images found
            if not reference_images:
                logger.warning(f"No reference images found for {target_file}, skipping")
                continue
                
            data.append({
                "file": target_path.as_posix(),
                "caption": caption_path.as_posix(),
                "references": reference_images
            })
        
        if not data:
            raise ValueError(f"No valid data found in {root} with reference patterns {reference_suffixes}")
            
        logger.info(f"Found {len(data)} {dataset_type}s with reference images")
        
        # Create dataset
        data = datasets.Dataset.from_list(data)
        
        # Cast to proper type
        if dataset_type in ["video", "video_references"]:
            data = data.rename_column("file", "video")
            data = data.cast_column("video", datasets.Video())
        else:
            data = data.rename_column("file", "image")
            data = data.cast_column("image", datasets.Image(mode="RGB"))
        
        # Load captions from files
        def _load_caption(sample):
            sample["caption"] = _read_caption_from_file(sample["caption"])
            return sample
        
        data = data.map(_load_caption)
        
        self._data = data.to_iterable_dataset()
        self._sample_index = 0
        self._precomputable_once = len(data) <= MAX_PRECOMPUTABLE_ITEMS_LIMIT
    
    def _get_data_iter(self):
        if self._sample_index == 0:
            return iter(self._data)
        return iter(self._data.skip(self._sample_index))
    
    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_index += 1
                yield sample
                
            if not self.infinite:
                logger.warning(f"Dataset ({self.__class__.__name__}={self.root}) has run out of data")
                break
            else:
                self._sample_index = 0
    
    def load_state_dict(self, state_dict):
        self._sample_index = state_dict["sample_index"]
    
    def state_dict(self):
        return {"sample_index": self._sample_index}


def initialize_reference_dataset(
    dataset_name_or_root: str,
    reference_suffixes: List[str] = None,
    dataset_type: str = "video",
    infinite: bool = False,
) -> torch.utils.data.IterableDataset:
    """Initialize a reference dataset from a local directory 
    
    Args:
        dataset_name_or_root: Path to local dir
        reference_suffixes: List of suffixes to identify reference images
        dataset_type: Type of dataset ("video", "video_references", or "image")
        infinite: Whether to loop the dataset infinitely
        
    Returns:
        An iterable dataset that pairs videos/images with reference images
    """
    assert dataset_type in ["image", "video", "video_references"], "Dataset type must be 'image', 'video', or 'video_references'"
    
    # Use local directory
    return PatternReferenceDataset(
        dataset_name_or_root, 
        reference_suffixes=reference_suffixes,
        dataset_type=dataset_type, 
        infinite=infinite
    )


def generate_video_resolution_buckets(
    data_root: str,
    video_resolutions: List[List[int]],
    reference_suffixes: List[str] = None,
) -> List[Tuple[int, int, int]]:
    """Generate video_resolution_buckets by collecting frame counts from dataset
    
    Args:
        data_root: Path to dataset directory
        video_resolutions: List of [height, width] pairs
        reference_suffixes: List of suffixes to identify reference images
        
    Returns:
        List of (frame_count, height, width) tuples for all videos in dataset
    """
    logger.info(f"Generating video_resolution_buckets for dataset in {data_root}")
    
    # Validate input
    if not video_resolutions:
        raise ValueError("At least one video resolution must be provided")
    
    # Find all video files in the dataset
    video_files = []
    for ext in constants.SUPPORTED_VIDEO_FILE_EXTENSIONS:
        video_files.extend(find_files(data_root, f"*.{ext}", depth=0))
    
    if not video_files:
        raise ValueError(f"No video files found in {data_root}")
    
    logger.info(f"Found {len(video_files)} video files in {data_root}")
    
    # Get frame counts for all videos
    frame_counts = set()
    for video_file in tqdm(video_files, desc="Scanning videos for frame counts"):
        try:
            video = load_video(video_file)
            if isinstance(video, torch.Tensor):
                frame_count = video.shape[0]
            else:
                # Handle other possible return types
                frame_count = len(video)
            frame_counts.add(frame_count)
            logger.info(f"Video {video_file} has {frame_count} frames")
        except Exception as e:
            logger.warning(f"Failed to load video {video_file}: {e}")
    
    if not frame_counts:
        raise ValueError("Could not determine frame counts from dataset")
    
    logger.info(f"Found {len(frame_counts)} unique frame counts: {sorted(frame_counts)}")
    
    # Generate buckets by combining each resolution with each frame count
    buckets = []
    for resolution in video_resolutions:
        if len(resolution) != 2:
            raise ValueError(f"Each resolution must be [height, width], got {resolution}")
        
        height, width = resolution
        for frame_count in sorted(frame_counts):
            buckets.append((frame_count, height, width))
    
    logger.info(f"Generated {len(buckets)} video_resolution_buckets: {buckets}")
    return buckets