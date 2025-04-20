import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed.checkpoint.stateful
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.video_processor import VideoProcessor
from PIL import Image

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.processors import ProcessorMixin
from finetrainers.typing import ArtifactType

from .config import E2VType, ElementConfig, FrameConditioningType


logger = get_logger()


class VAEPathwayProcessor(ProcessorMixin):
    """Processor for the VAE spatial pathway."""
    
    def __init__(self, output_names=None, input_names=None, config=None, device=None):
        super().__init__()
        self.output_names = output_names or ["vae_output"]
        self.input_names = input_names or {}
        self.config = config
        self.device = device
        
    def forward(self, image=None, video=None, element_config=None, **kwargs):
        """Process image/video through VAE pathway.
        
        Args:
            image: Optional image tensor (B, C, H, W)
            video: Optional video tensor (B, F, C, H, W)
            element_config: Configuration for this element
            
        Returns:
            Dictionary with processed VAE output
        """
        # 1. Get configuration with element-specific overrides
        config = dict(self.config)
        if element_config and "vae" in element_config:
            config.update(element_config["vae"])
        
        # 2. Preprocess image/video
        processed = self._preprocess_input(image, video, config)
        
        # 3. Apply repetition based on config
        repeated = self._apply_repetition(processed, config.get("repeat", 1))
        
        # Store result for later concatenation
        result = {
            "latents": repeated,
            "position": config.get("position", 0),
            "frames": repeated.shape[1] if len(repeated.shape) > 3 else 1
        }
        
        return {self.output_names[0]: result}
    
    def _preprocess_input(self, image, video, config):
        """Preprocess input image or video."""
        if image is not None:
            # For a single image, add a frame dimension
            if len(image.shape) == 4:  # (B, C, H, W)
                return image.unsqueeze(2)  # (B, C, 1, H, W)
            return image
        elif video is not None:
            return video
        else:
            raise ValueError("Either image or video must be provided")
    
    def _apply_repetition(self, video, repeat):
        """Apply repetition to create the mini-video of reference frames."""
        if repeat <= 1:
            return video
        
        if len(video.shape) == 5:  # (B, C, F, H, W)
            # Determine which frames to repeat
            frame_dim = 2
            frames = video.shape[frame_dim]
            
            # Handle single frame case
            if frames == 1:
                return torch.cat([video] * repeat, dim=frame_dim)
            
            # For multiple frames, repeat each frame as specified
            repeated_frames = []
            for i in range(frames):
                frame = video[:, :, i:i+1, :, :]
                repeated_frames.append(torch.cat([frame] * repeat, dim=frame_dim))
            
            return torch.cat(repeated_frames, dim=frame_dim)
        else:
            return video


class CLIPPathwayProcessor(ProcessorMixin):
    """Processor for the CLIP semantic pathway."""
    
    def __init__(self, output_names=None, input_names=None, config=None, device=None, clip_processor=None):
        super().__init__()
        self.output_names = output_names or ["clip_output"]
        self.input_names = input_names or {}
        self.config = config
        self.device = device
        self.clip_processor = clip_processor
        
    def forward(self, image=None, element_config=None, **kwargs):
        """Process image through CLIP pathway.
        
        Args:
            image: Image tensor (B, C, H, W)
            element_config: Configuration for this element
            
        Returns:
            Dictionary with processed CLIP features
        """
        # 1. Get configuration with element-specific overrides
        config = dict(self.config)
        if element_config and "clip" in element_config:
            if isinstance(element_config["clip"], dict):
                config.update(element_config["clip"])
            elif not element_config["clip"]:
                # CLIP pathway disabled for this element
                return {self.output_names[0]: None}
        
        # 2. Preprocess image
        processed = self._preprocess_input(image, config)
        
        # 3. Run CLIP encoder via the clip_processor
        if self.clip_processor is not None:
            clip_inputs = {"image": processed}
            clip_outputs = self.clip_processor(**clip_inputs)
            features = clip_outputs.get("encoder_hidden_states", None)
            
            return {self.output_names[0]: features}
        else:
            # If no CLIP processor, just return preprocessed image
            return {self.output_names[0]: processed}
    
    def _preprocess_input(self, image, config):
        """Preprocess image for CLIP."""
        if image is None:
            raise ValueError("Image must be provided for CLIP processing")
            
        # Check for direct preprocessor configuration
        preprocess_type = config.get("preprocess", config.get("default_preprocess", "center_crop"))
        resolution = config.get("resolution", [224, 224])
        
        # Apply preprocessing based on type
        if preprocess_type == "center_crop":
            return FF.center_crop_image(image, resolution)
        elif preprocess_type == "resize":
            return FF.resize_image(image, resolution)
        elif preprocess_type == "pad_white":
            return FF.pad_image(image, resolution, padding_color=1.0)
        elif preprocess_type == "letterbox":
            return FF.letterbox_image(image, resolution)
        else:
            # Default to center crop
            return FF.center_crop_image(image, resolution)


def apply_frame_conditioning_on_latents(
    latents: torch.Tensor,
    expected_num_frames: int,
    channel_dim: int,
    frame_dim: int,
    frame_conditioning_type: str,
    frame_conditioning_index: Optional[int] = None,
    concatenate_mask: bool = False,
) -> torch.Tensor:
    """Apply frame conditioning on latents, similar to control training."""
    num_frames = latents.size(frame_dim)
    mask = torch.zeros_like(latents)

    if frame_conditioning_type == FrameConditioningType.INDEX:
        frame_index = min(frame_conditioning_index or 0, num_frames - 1)
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = frame_index
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.PREFIX:
        frame_index = random.randint(1, num_frames)
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = slice(0, frame_index)  # Keep frames 0 to frame_index-1
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.RANDOM:
        # Zero or more random frames to keep
        num_frames_to_keep = random.randint(1, num_frames)
        frame_indices = random.sample(range(num_frames), num_frames_to_keep)
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = frame_indices
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.FIRST_AND_LAST:
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = 0
        mask[tuple(indexing)] = 1
        indexing[frame_dim] = num_frames - 1
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.FULL:
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = slice(0, num_frames)
        mask[tuple(indexing)] = 1

    # Handle padding/truncation to match expected number of frames
    if latents.size(frame_dim) >= expected_num_frames:
        slicing = [slice(None)] * latents.ndim
        slicing[frame_dim] = slice(expected_num_frames)
        latents = latents[tuple(slicing)]
        mask = mask[tuple(slicing)]
    else:
        pad_size = expected_num_frames - num_frames
        pad_shape = list(latents.shape)
        pad_shape[frame_dim] = pad_size
        padding = latents.new_zeros(pad_shape)
        latents = torch.cat([latents, padding], dim=frame_dim)
        mask = torch.cat([mask, padding], dim=frame_dim)

    if concatenate_mask:
        latents = torch.cat([latents, mask], dim=channel_dim)

    return latents


class IterableE2VDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    """Dataset wrapper for E2V training."""
    
    def __init__(self, dataset, config, device=None, clip_processor=None, vae=None):
        super().__init__()
        
        self.dataset = dataset
        self.config = config
        self.device = device
        self.clip_processor = clip_processor
        self.vae = vae  # VAE for encoding reference images
        
        
        # Initialize processors
        self.processors = {}
        if "vae" in config.get("processors", {}):
            self.processors["vae"] = VAEPathwayProcessor(
                output_names=["vae_output"],
                config=config["processors"]["vae"],
                device=device
            )
        
        if config.get("e2v_type") in [E2VType.CLIP, E2VType.DUAL]:
            if "clip" in config.get("processors", {}):
                self.processors["clip"] = CLIPPathwayProcessor(
                    output_names=["clip_output"],
                    config=config["processors"]["clip"],
                    device=device,
                    clip_processor=clip_processor
                )
        
        # Create element lookup - assume elements are dictionaries
        self.elements = {}
        for elem in config.get("elements", []):
            self.elements[elem["name"]] = elem
            
        logger.info(f"Initialized IterableE2VDataset with {len(self.elements)} elements")
    
    def __iter__(self):
        logger.info("Starting IterableE2VDataset")
        for data in iter(self.dataset):
            try:
                # Basic logging to understand dataset structure
                keys = list(data.keys())
                logger.info(f"Dataset item keys: {keys}")
                
                if "video" in data:
                    logger.info(f"Video shape: {data['video'].shape}")
                
                # Find element files based on dataset item
                element_files = self._find_element_files(data)
                
                # Skip items where required elements are missing
                if not element_files:
                    required_elements = [elem["name"] for elem in self.config.get("elements", []) 
                                        if elem.get("required", False)]
                    if required_elements:
                        logger.warning(f"Skipping item because required elements not found: {required_elements}")
                        continue
                
                # Load element images
                element_data = self._load_elements(element_files)
                
                # If loading failed, skip this item
                if not element_data:
                    logger.warning("No elements could be loaded, skipping item")
                    continue
                
                # Process elements through VAE and CLIP pathways
                processed_data = self._process_elements(data, element_data)
                
                # Combine all pathways into final output
                combined_data = self._combine_pathways(data, processed_data)
                
                yield combined_data
            except Exception as e:
                logger.error(f"Error processing dataset item: {e}")
                # Skip this item and continue
                continue
    
    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)
    
    def state_dict(self):
        return self.dataset.state_dict()
    
    def _find_element_files(self, data):
        """Process reference images from VideoReferenceImagesDataset."""
        element_files = {}
        
        # Log what dataset provides
        logger.info(f"Dataset item keys: {list(data.keys())}")
        
        # Check if we have reference images from our VideoReferenceImagesDataset
        if "images" in data:
            import os
            
            logger.info(f"Found {len(data['images'])} reference images in dataset item")
            
            # Process each reference image
            for image_path in data["images"]:
                # Get the basename to match with suffixes
                filename = os.path.basename(image_path)
                
                # Try to match with one of our configured element types
                for element_name, element_config in self.elements.items():
                    # Check if this image matches one of the element's suffixes
                    for config_suffix in element_config["suffixes"]:
                        if filename.endswith(config_suffix):
                            element_files[element_name] = {
                                "path": image_path,
                                "config": element_config
                            }
                            logger.info(f"Found {element_name} reference image: {image_path}")
                            break
            
            if not element_files:
                logger.warning("No matching reference images found in dataset item")
        else:
            logger.warning("No reference images ('images' key) found in dataset item")
        
        # All processing done with direct file paths from dataset
        
        # Log results
        if element_files:
            logger.info(f"Found {len(element_files)} element files")
        else:
            logger.warning(f"No element files found in dataset item")
            # Log what elements we were looking for
            element_types = [name for name in self.elements]
            logger.info(f"Was looking for element types: {element_types}")
        
        return element_files
    
    def _load_elements(self, element_files):
        """Load element images from files."""
        element_data = {}
        
        # Load each element
        for element_name, file_info in element_files.items():
            try:
                # Load image
                image_path = file_info["path"]
                
                # Load and process image
                element_img = Image.open(image_path).convert("RGB")
                
                # Convert to tensor
                video_processor = VideoProcessor()
                element_tensor = video_processor.preprocess(element_img)
                
                # Store in element data
                element_data[element_name] = {
                    "image": element_tensor,
                    "config": file_info["config"]
                }
            except Exception as e:
                logger.error(f"Error loading element {element_name}: {e}")
                # Skip this element if it fails to load
                continue
        
        return element_data
    
    def _process_elements(self, data, element_data):
        """Process each element through each pathway."""
        results = {"vae": {}, "clip": {}}
        
        # Process each element through each pathway
        for element_name, element_info in element_data.items():
            element_image = element_info["image"]
            element_config = element_info["config"]
            
            # Process through each pathway
            for processor_name, processor in self.processors.items():
                # Skip if the configuration doesn't enable this processor for this element
                if processor_name == "clip" and not element_config.get("clip", False):
                    continue
                
                # Process the element
                if processor_name == "vae":
                    result = processor(image=element_image, element_config=element_config)
                elif processor_name == "clip":
                    result = processor(image=element_image, element_config=element_config)
                else:
                    continue
                
                # Store result if pathway is enabled and returned a valid result
                output_name = processor.output_names[0]
                if result[output_name] is not None:
                    results[processor_name][element_name] = result[output_name]
        
        return results
    
    def _combine_pathways(self, data, processed_data):
        """Combine results from all pathways."""
        result_data = dict(data)
        
        # Combine VAE pathway results
        if "vae" in processed_data and processed_data["vae"]:
            vae_results = list(processed_data["vae"].values())
            
            # Sort by position
            vae_results.sort(key=lambda x: x["position"])
            
            # Concatenate along frame dimension
            vae_latents = torch.cat([r["latents"] for r in vae_results], dim=2)
            
            # Create mask
            frame_dim = 2
            channel_dim = 1
            num_frames = vae_latents.shape[frame_dim]
            mask = torch.zeros_like(vae_latents[:, :1])  # Take only first channel for mask
            mask[:, :, :sum(r["frames"] for r in vae_results)] = 1
            
            # Apply frame conditioning
            frame_cond_type = self.config.get("frame_conditioning_type", FrameConditioningType.FULL)
            frame_cond_index = self.config.get("frame_conditioning_index", 0)
            concatenate_mask = self.config.get("frame_conditioning_concatenate_mask", True)
            
            # Apply frame conditioning to match expected video frames
            expected_frames = data["video"].shape[1] if "video" in data else None
            if expected_frames:
                vae_latents = apply_frame_conditioning_on_latents(
                    vae_latents,
                    expected_frames,
                    channel_dim,
                    frame_dim,
                    frame_cond_type,
                    frame_cond_index,
                    False  # We'll handle mask concatenation separately
                )
                
                if concatenate_mask:
                    # Match mask shape to conditioned latents
                    mask = mask[:, :, :vae_latents.shape[frame_dim]]
                    
                    # Concatenate mask with latents
                    vae_latents = torch.cat([vae_latents, mask], dim=channel_dim)
            
            # Add to result
            result_data["e2v_vae_latents"] = vae_latents
        
        # Combine CLIP pathway results
        if "clip" in processed_data and processed_data["clip"]:
            clip_embeddings = list(processed_data["clip"].values())
            
            # Concatenate along sequence dimension (dim=1)
            if clip_embeddings and all(e is not None for e in clip_embeddings):
                # Assume clip_embeddings is a list of tensors of shape [batch, seq_len, hidden_dim]
                clip_combined = torch.cat(clip_embeddings, dim=1)
                result_data["e2v_clip_embeddings"] = clip_combined
        
        return result_data


class ValidationE2VDataset(IterableE2VDataset):
    """Validation dataset for E2V training."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __iter__(self):
        logger.info("Starting ValidationE2VDataset")
        for data in iter(self.dataset):
            try:
                # Find element files
                element_files = self._find_element_files(data)
                
                # Process elements
                element_data = self._load_elements(element_files)
                processed_data = self._process_elements(data, element_data)
                combined_data = self._combine_pathways(data, processed_data)
                
                # For validation, also include the original element files
                combined_data["element_files"] = element_files
                
                yield combined_data
            except Exception as e:
                logger.error(f"Error processing validation dataset item: {e}")
                continue