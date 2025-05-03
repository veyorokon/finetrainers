import itertools
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed.checkpoint.stateful
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image

import finetrainers.functional as FF
from finetrainers.functional.image import letterbox_image
from finetrainers.logging import get_logger
from finetrainers.processors import CannyProcessor, CopyProcessor
from finetrainers.trainer.control_trainer.data import (ControlType,
                                                       FrameConditioningType,
                                                       IterableControlDataset)

logger = get_logger()


class IterableReferenceDataset(IterableControlDataset):
    """Dataset for reference conditioning that adds reference image processing.
    
    This extends IterableControlDataset to also handle reference images for
    CLIP embedding conditioning alongside control images for latent conditioning.
    """
    def __init__(
        self, 
        dataset: torch.utils.data.IterableDataset,
        control_type: str, 
        reference_config: Dict[str, Any] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__(dataset, control_type, device)
        
        if reference_config is None:
            raise ValueError("IterableReferenceDataset requires reference_config")
        self.reference_config = reference_config

        logger.info("Initialized IterableReferenceDataset with config:")
        logger.info(f"  VAE Resolution: {self.reference_config['vae_resolution']}")
        logger.info(f"  CLIP Resolution: {self.reference_config['clip_resolution']}")
        logger.info(f"  Reference Order: {self.reference_config['reference_order']}")
        logger.info(f"  Repeat Frames: {self.reference_config['repeat_frames']}")
    
    def __iter__(self):
        logger.info("===== Starting IterableReferenceDataset =====")
        logger.info(f"Reference config: {self.reference_config}")
        
        # First check the source dataset
        source_iter = iter(self.dataset)
        try:
            first_item = next(source_iter)
            logger.info(f"First item from source dataset has keys: {list(first_item.keys())}")
            if "references" in first_item:
                logger.info(f"First item has references with keys: {list(first_item['references'].keys())}")
                for ref_key, ref_path in first_item['references'].items():
                    logger.info(f"  Reference {ref_key}: {ref_path}")
            else:
                logger.warning("First item does not have 'references' key - check dataset configuration!")
                
            # Put the item back (restore the iterator state)
            source_iter = itertools.chain([first_item], source_iter)
        except StopIteration:
            logger.error("Source dataset is empty! No data to process.")
            source_iter = iter([])  # Empty iterator
            
        # Now process normally
        for data in source_iter:
            # Log what we initially received from the dataset
            logger.info(f"IterableReferenceDataset processing item with keys: {list(data.keys())}")
            
            # Process reference images only for CLIP embedding
            if "references" in data:
                logger.info(f"Processing references: {list(data['references'].keys())}")
                clip_images = []
                vae_images = []
                
                # Get config values
                vae_resolution = self.reference_config["vae_resolution"]
                clip_resolution = self.reference_config["clip_resolution"]
                reference_order = self.reference_config["reference_order"]
                repeat_frames = self.reference_config["repeat_frames"]
                
                # Process reference images in specified order
                for idx, ref_type in enumerate(reference_order):
                    if ref_type in data["references"]:
                        # Load the reference image
                        ref_path = data["references"][ref_type]
                        logger.info(f"Loading reference '{ref_type}' from {ref_path}")
                        ref_image = load_image(ref_path)
                        
                        # Get repetition count (or default to 1)
                        repeat = repeat_frames[idx] if idx < len(repeat_frames) else 1
                        logger.info(f"Reference '{ref_type}' will repeat {repeat} times")
                        
                        # Process for VAE storage (no longer creating control video here)
                        vae_image = letterbox_image(
                            ref_image, vae_resolution
                        )
                        vae_images.append({"image": vae_image, "repeat": repeat})
                        
                        # Process for CLIP
                        clip_image = letterbox_image(
                            ref_image, clip_resolution
                        )
                        clip_images.append(clip_image)
                
                # Store the processed images for reference path
                # Control video creation now happens in the ReferenceToControlProcessor
                data["vae_references"] = vae_images
                data["clip_references"] = clip_images
                logger.info(f"Processed {len(vae_images)} reference images into vae_references")
                logger.info(f"Processed {len(clip_images)} reference images into clip_references")
            else:
                logger.info("No references found in data")
            
            # Now process control images/videos as in parent class
            logger.info("Running control processors")
            control_augmented_data = self._run_control_processors(data)
            
            # Log what we're yielding back
            logger.info(f"IterableReferenceDataset yielding data with keys: {list(control_augmented_data.keys())}")
            
            # Check for references specifically
            if "references" in control_augmented_data:
                logger.info(f"Original references present with keys: {list(control_augmented_data['references'].keys())}")
            
            # Check for vae_references
            if "vae_references" in control_augmented_data:
                logger.info(f"vae_references present with {len(control_augmented_data['vae_references'])} items")
                # Log details of each reference
                for i, ref in enumerate(control_augmented_data["vae_references"]):
                    img_type = type(ref["image"]).__name__
                    repeat = ref["repeat"]
                    img_info = f"size={ref['image'].size}" if hasattr(ref["image"], "size") else "no size"
                    logger.info(f"  vae_reference {i}: type={img_type}, repeat={repeat}, {img_info}")
            
            # Log clip references as well
            if "clip_references" in control_augmented_data:
                logger.info(f"clip_references present with {len(control_augmented_data['clip_references'])} items")
                # Log type of each
                for i, ref in enumerate(control_augmented_data["clip_references"]):
                    img_type = type(ref).__name__
                    img_info = f"size={ref.size}" if hasattr(ref, "size") else "no size"
                    logger.info(f"  clip_reference {i}: type={img_type}, {img_info}")
                
            yield control_augmented_data

    def _run_control_processors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Use parent implementation
        return super()._run_control_processors(data)


class ValidationReferenceDataset(torch.utils.data.IterableDataset):
    """Validation dataset for reference conditioning."""
    
    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        control_type: str,
        reference_config: Dict[str, Any] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.dataset = dataset
        self.control_type = control_type
        self.device = device
        self._video_processor = VideoProcessor()
        
        if reference_config is None:
            raise ValueError("ValidationReferenceDataset requires reference_config")
        self.reference_config = reference_config
        
        self.control_processors = []
        if control_type == ControlType.CANNY:
            self.control_processors.append(
                CannyProcessor(["control_output"], input_names={"image": "input", "video": "input"}, device=device)
            )
        elif control_type == ControlType.NONE:
            self.control_processors.append(
                CopyProcessor(["control_output"], input_names={"image": "input", "video": "input"})
            )

        logger.info("Initialized ValidationReferenceDataset")
    
    def __iter__(self):
        logger.info("Starting ValidationReferenceDataset")
        for data in iter(self.dataset):
            # Process reference images only for CLIP embedding
            if "references" in data:
                clip_images = []
                vae_images = []
                
                # Get config values
                vae_resolution = self.reference_config["vae_resolution"]
                clip_resolution = self.reference_config["clip_resolution"]
                reference_order = self.reference_config["reference_order"]
                repeat_frames = self.reference_config["repeat_frames"]
                
                # Process reference images in specified order
                for idx, ref_type in enumerate(reference_order):
                    if ref_type in data["references"]:
                        # Load the reference image
                        ref_path = data["references"][ref_type]
                        ref_image = load_image(ref_path)
                        
                        # Get repetition count (or default to 1)
                        repeat = repeat_frames[idx] if idx < len(repeat_frames) else 1
                        
                        # Process for VAE storage (no longer creating control video here)
                        vae_image = letterbox_image(
                            ref_image, vae_resolution
                        )
                        vae_images.append({"image": vae_image, "repeat": repeat})
                        
                        # Process for CLIP
                        clip_image = letterbox_image(
                            ref_image, clip_resolution
                        )
                        clip_images.append(clip_image)
                
                # Store the processed images for reference path
                # Control video creation is handled by the ReferenceToControlProcessor
                data["vae_references"] = vae_images
                data["clip_references"] = clip_images
                logger.info(f"ValidationReferenceDataset: Processed {len(vae_images)} references into vae_references")
            
            # Process control images/videos
            control_augmented_data = self._run_control_processors(data)
            
            yield control_augmented_data
    
    def _run_control_processors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Same implementation as ValidationControlDataset
        if self.control_type == ControlType.CUSTOM:
            return data
            
        # These are already expected to be tensors
        if "control_image" in data or "control_video" in data:
            return data
            
        shallow_copy_data = dict(data.items())
        is_image_control = "image" in shallow_copy_data
        is_video_control = "video" in shallow_copy_data
        
        if (is_image_control + is_video_control) != 1:
            raise ValueError("Exactly one of 'image' or 'video' should be present in the data.")
            
        for processor in self.control_processors:
            result = processor(**shallow_copy_data)
            result_keys = set(result.keys())
            repeat_keys = result_keys.intersection(shallow_copy_data.keys())
            
            if repeat_keys:
                logger.warning(
                    f"Processor {processor.__class__.__name__} returned keys that already exist in "
                    f"conditions: {repeat_keys}. Overwriting the existing values, but this may not "
                    f"be intended. Please rename the keys in the processor to avoid conflicts."
                )
                
            shallow_copy_data.update(result)
            
        if "control_output" in shallow_copy_data:
            # Normalize to [-1, 1] range
            control_output = shallow_copy_data.pop("control_output")
            
            if torch.is_tensor(control_output):
                control_output = FF.normalize(control_output, min=-1.0, max=1.0)
                ndim = control_output.ndim
                
                assert 3 <= ndim <= 5, "Control output should be at least ndim=3 and less than or equal to ndim=5"
                
                if ndim == 5:
                    control_output = self._video_processor.postprocess_video(control_output, output_type="pil")
                else:
                    if ndim == 3:
                        control_output = control_output.unsqueeze(0)
                    control_output = self._video_processor.postprocess(control_output, output_type="pil")[0]
                    
            key = "control_image" if is_image_control else "control_video"
            shallow_copy_data[key] = control_output
            
        return shallow_copy_data
    

def apply_reference_frame_conditioning(
    latents: torch.Tensor,
    expected_num_frames: int,
    frame_conditioning_type: str,
    frame_conditioning_index: Optional[int] = None,
    channel_dim: int = 1,
    frame_dim: int = 2,
    concatenate_mask: bool = True,
) -> torch.Tensor:
    """
    Apply frame conditioning for reference model with optional A2-style single-channel mask.
    
    This is a simplified implementation that doesn't rely on apply_frame_conditioning_on_latents.
    It directly handles the frame conditioning and mask creation in a single function.
    
    Args:
        latents: Control latents to condition
        expected_num_frames: Number of frames to match (output frame count)
        frame_conditioning_type: Type of conditioning ("index", "full", etc.)
        frame_conditioning_index: Index for index-based conditioning
        channel_dim: Dimension for channels
        frame_dim: Dimension for frames
        concatenate_mask: Whether to concatenate a single-channel mask
        
    Returns:
        Conditioned latents, optionally with single-channel mask concatenated
    """
    # Get original frame count and create result tensor of expected size
    original_frames = latents.size(frame_dim)
    
    # Log input and expected shapes
    logger.info(f"Reference conditioning: input shape={latents.shape}, frames={original_frames}, " + 
              f"expected_frames={expected_num_frames}")
    
    # Create result tensor with correct size (padded to expected_num_frames)
    result_shape = list(latents.shape)
    result_shape[frame_dim] = expected_num_frames
    result = torch.zeros(result_shape, device=latents.device, dtype=latents.dtype)
    
    # Find frames to keep based on conditioning type
    if frame_conditioning_type == "index":
        # Only keep a single frame specified by index
        frame_index = min(frame_conditioning_index or 0, original_frames - 1)
        kept_indices = [frame_index]
    elif frame_conditioning_type == "first_and_last":
        # Keep first and last frames
        kept_indices = [0, original_frames - 1]
    elif frame_conditioning_type == "full":
        # Keep all original frames
        kept_indices = list(range(original_frames))
    else:
        # Default to keeping all frames
        kept_indices = list(range(original_frames))
    
    # Limit to expected frame count
    kept_indices = kept_indices[:expected_num_frames]
    logger.info(f"Keeping frames: {kept_indices}")
    
    # Create the mask tensor (1 where frames are kept, 0 elsewhere)
    mask_shape = list(result.shape)
    mask_shape[channel_dim] = 1  # Single channel for mask
    mask = torch.zeros(mask_shape, device=latents.device, dtype=latents.dtype)
    
    # Fast way to copy multiple frames to result and set mask
    if kept_indices:
        # Convert kept_indices to tensor for indexing
        kept_indices_tensor = torch.tensor(kept_indices, device=latents.device)
        
        # Copy selected frames from latents to result
        # This selects frames at positions in kept_indices from latents
        # and places them at the beginning of result
        result[:, :, :len(kept_indices)] = latents[:, :, kept_indices_tensor]
        
        # Set mask to 1 for kept frames
        mask[:, :, :len(kept_indices)] = 1
        
        logger.info(f"Copied {len(kept_indices)} frames and set mask values")
    
    # If mask is not needed, return result directly
    if not concatenate_mask:
        return result
        
    # Log mask statistics for debugging
    mask_min = mask.min().item()
    mask_max = mask.max().item()
    mask_mean = mask.mean().item()
    mask_nonzero = (mask > 0).float().sum().item()
    logger.info(f"Mask stats: min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}, " +
              f"non-zero={mask_nonzero} out of {mask.numel()}")
    
    # Create 4 mask channels (A2 model expects 4 mask + 16 content + 16 conditioning = 36 channels)
    num_mask_channels = 4
    
    # Expand mask to 4 channels directly using repeat
    masks = mask.repeat(1, num_mask_channels, 1, 1, 1)
    
    # Ensure masks require gradients (for compatibility with previous code)
    masks = masks.requires_grad_(True)
    
    # Concatenate masks with result (masks first, then content)
    combined = torch.cat([masks, result], dim=channel_dim)
    
    logger.info(f"Applied A2-style reference conditioning: masks shape={masks.shape}, " +
               f"combined shape={combined.shape}")
    
    return combined