import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed.checkpoint.stateful
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image
import torchvision.transforms as transforms

import finetrainers.functional as FF
from finetrainers.logging import get_logger
from finetrainers.processors import CannyProcessor, CopyProcessor
from finetrainers.trainer.control_trainer.data import (ControlType,
                                                       FrameConditioningType,
                                                       IterableControlDataset)

logger = get_logger()


def _crop_and_resize_pad(image, height, width, resize_mode="bicubic"):
    """Center crop and resize image with padding to maintain aspect ratio."""
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL for processing
        if image.dim() == 3:  # [C, H, W]
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 127.5 + 127.5).astype("uint8"))
        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")
    
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Determine aspect ratio
    target_ratio = width / height
    orig_ratio = orig_width / orig_height
    
    if orig_ratio > target_ratio:
        # Image is wider than target ratio
        new_width = int(orig_height * target_ratio)
        new_height = orig_height
        left = (orig_width - new_width) // 2
        image = image.crop((left, 0, left + new_width, new_height))
    else:
        # Image is taller than target ratio
        new_width = orig_width
        new_height = int(orig_width / target_ratio)
        top = (orig_height - new_height) // 2
        image = image.crop((0, top, new_width, top + new_height))
    
    # Resize to target dimensions
    image = image.resize((width, height), getattr(Image, resize_mode.upper()))
    return image


def _crop_and_resize(image, height, width, resize_mode="bicubic"):
    """Resize image without padding, allowing aspect ratio change."""
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL for processing
        if image.dim() == 3:  # [C, H, W]
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 127.5 + 127.5).astype("uint8"))
        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")
    
    # Resize to target dimensions
    image = image.resize((width, height), getattr(Image, resize_mode.upper()))
    return image


def pil_to_tensor(image):
    """Convert PIL image to normalized tensor in range [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)


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
        
        self.reference_config = reference_config or {
            "vae_resolution": [854, 480],
            "clip_resolution": [512, 512],
            "reference_order": ["object", "background"],
            "repeat_frames": [1, 4]
        }

        logger.info("Initialized IterableReferenceDataset with config:")
        logger.info(f"  VAE Resolution: {self.reference_config['vae_resolution']}")
        logger.info(f"  CLIP Resolution: {self.reference_config['clip_resolution']}")
        logger.info(f"  Reference Order: {self.reference_config['reference_order']}")
        logger.info(f"  Repeat Frames: {self.reference_config['repeat_frames']}")
    
    def __iter__(self):
        logger.info("Starting IterableReferenceDataset")
        for data in iter(self.dataset):
            # Process reference images first, then create control_video from them
            if "references" in data:
                vae_images = []
                clip_images = []
                processed_vae_tensors = []
                
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
                        
                        # Process for VAE
                        vae_image = _crop_and_resize_pad(
                            ref_image,
                            height=vae_resolution[1],
                            width=vae_resolution[0]
                        )
                        vae_images.append({"image": vae_image, "repeat": repeat})
                        
                        # Convert to tensor for video creation
                        vae_tensor = pil_to_tensor(vae_image)
                        processed_vae_tensors.append((vae_tensor, repeat))
                        
                        # Process for CLIP
                        clip_image = _crop_and_resize_pad(
                            ref_image,
                            height=clip_resolution[1],
                            width=clip_resolution[0]
                        )
                        clip_images.append(clip_image)
                
                # Create control_video from reference images
                if processed_vae_tensors and ("control_image" not in data and "control_video" not in data):
                    # Create a sequence of frames with specified repetitions
                    frames = []
                    for tensor, repeat_count in processed_vae_tensors:
                        frames.extend([tensor] * repeat_count)
                    
                    if frames:
                        # Stack frames to create video [T, C, H, W]
                        control_video = torch.stack(frames, dim=0)
                        # Add batch dimension [B, T, C, H, W]
                        control_video = control_video.unsqueeze(0)
                        # Permute to [B, C, T, H, W] format for VAE
                        control_video = control_video.permute(0, 2, 1, 3, 4)
                        data["control_video"] = control_video
                
                # Store the processed images for reference path
                data["vae_references"] = vae_images
                data["clip_references"] = clip_images
            
            # Now process control images/videos as in parent class
            control_augmented_data = self._run_control_processors(data)
            
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
        
        self.reference_config = reference_config or {
            "vae_resolution": [854, 480],
            "clip_resolution": [512, 512],
            "reference_order": ["object", "background"],
            "repeat_frames": [1, 4]
        }
        
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
            # Process reference images first, then create control inputs
            if "references" in data:
                vae_images = []
                clip_images = []
                processed_vae_tensors = []
                
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
                        
                        # Process for VAE
                        vae_image = _crop_and_resize_pad(
                            ref_image,
                            height=vae_resolution[1],
                            width=vae_resolution[0]
                        )
                        vae_images.append({"image": vae_image, "repeat": repeat})
                        
                        # Convert to tensor for video creation
                        vae_tensor = pil_to_tensor(vae_image)
                        processed_vae_tensors.append((vae_tensor, repeat))
                        
                        # Process for CLIP
                        clip_image = _crop_and_resize_pad(
                            ref_image,
                            height=clip_resolution[1],
                            width=clip_resolution[0]
                        )
                        clip_images.append(clip_image)
                
                # Create control_video from reference images
                if processed_vae_tensors and ("control_image" not in data and "control_video" not in data):
                    # Create a sequence of frames with specified repetitions
                    frames = []
                    for tensor, repeat_count in processed_vae_tensors:
                        frames.extend([tensor] * repeat_count)
                    
                    if frames:
                        # Stack frames to create video [T, C, H, W]
                        control_video = torch.stack(frames, dim=0)
                        # Add batch dimension [B, T, C, H, W]
                        control_video = control_video.unsqueeze(0)
                        # Permute to [B, C, T, H, W] format for VAE
                        control_video = control_video.permute(0, 2, 1, 3, 4)
                        data["control_video"] = control_video
                
                # Store the processed images for reference path
                data["vae_references"] = vae_images
                data["clip_references"] = clip_images
            
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