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
            "frames": repeated.shape[2] if len(repeated.shape) > 3 else 1
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
    """Dataset wrapper for E2V (Elements-to-Video) training.
    
    This wrapper processes video datasets along with reference images to create
    the combined conditioning needed for E2V training. It follows the same pattern
    as other framework wrappers like IterableControlDataset.
    
    The wrapper's main functions:
    1. Identify and load reference images for each video
    2. Process references through VAE and CLIP pathways
    3. Create the specialized conditioning tensors
    4. Feed properly formatted data to the E2V training pipeline
    
    It coordinates all the processing needed for both spatial conditioning (VAE)
    and semantic conditioning (CLIP) used in the A2 approach.
    """
    
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
                    required_elements = []
                    for elem in self.config.get("elements", []):
                        if elem.get("required", False):
                            required_elements.append({
                                "name": elem["name"],
                                "suffixes": elem.get("suffixes", [])
                            })
                    
                    if required_elements:
                        # More detailed warning message with expected file patterns
                        element_details = [f"{e['name']} (patterns: {', '.join(e['suffixes'])})" for e in required_elements]
                        logger.warning(f"Skipping dataset item - required elements not found: {element_details}")
                        if "images" in data:
                            logger.warning(f"Available reference images: {data['images']}")
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
        """Process reference images from VideoReferenceImagesDataset.
        
        This method:
        1. Extracts reference image paths from the dataset item
        2. Matches them to the configured element types based on file suffixes
        3. Creates a mapping from element types to their corresponding image paths
        
        Args:
            data: Dataset item containing video tensor and reference image paths
            
        Returns:
            Dictionary mapping element types to their file info (path and config)
        """
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
                logger.warning("No reference images matched the configured element patterns")
                # Provide more detailed information about what was expected
                element_patterns = []
                for name, config in self.elements.items():
                    if "suffixes" in config:
                        element_patterns.append(f"{name}: {config['suffixes']}")
                
                logger.info(f"Expected element patterns: {element_patterns}")
                if "images" in data:
                    logger.info(f"Available reference images: {data['images']}")
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
        """Process elements through each pathway using batch processing.
        
        Groups elements by processor type and processes them in batches for efficiency.
        """
        results = {proc_name: {} for proc_name in self.processors}
        
        # Group elements by processor for batch processing
        processor_inputs = {proc_name: [] for proc_name in self.processors}
        processor_configs = {proc_name: [] for proc_name in self.processors}
        processor_elements = {proc_name: [] for proc_name in self.processors}
        
        # Map elements to processors
        for element_name, element_info in element_data.items():
            element_image = element_info["image"]
            element_config = element_info["config"]
            
            # Check which processors are configured for this element
            for proc_name, processor in self.processors.items():
                # Skip if the processor is explicitly disabled for this element
                if proc_name == "clip" and not element_config.get("clip", False):
                    continue
                
                processor_inputs[proc_name].append(element_image)
                processor_configs[proc_name].append(element_config)
                processor_elements[proc_name].append(element_name)
        
        # Process each processor type in batches
        for proc_name, processor in self.processors.items():
            if not processor_inputs[proc_name]:
                continue
            
            # Get batch size from processor config
            batch_size = self.config.get("processors", {}).get(proc_name, {}).get("batch_size", len(processor_inputs[proc_name]))
            
            # Process in batches if needed
            for batch_start in range(0, len(processor_inputs[proc_name]), batch_size):
                batch_end = min(batch_start + batch_size, len(processor_inputs[proc_name]))
                
                batch_inputs = processor_inputs[proc_name][batch_start:batch_end]
                batch_configs = processor_configs[proc_name][batch_start:batch_end]
                batch_elements = processor_elements[proc_name][batch_start:batch_end]
                
                # Process each element in the batch
                for i, (element_input, element_config, element_name) in enumerate(zip(batch_inputs, batch_configs, batch_elements)):
                    # Process the element based on processor type
                    if proc_name == "vae":
                        result = processor(image=element_input, element_config=element_config)
                    elif proc_name == "clip":
                        result = processor(image=element_input, element_config=element_config)
                    else:
                        continue
                    
                    # Store result if pathway is enabled and returned a valid result
                    output_name = processor.output_names[0]
                    if result[output_name] is not None:
                        results[proc_name][element_name] = result[output_name]
        
        return results
    
    def _combine_pathways(self, data, processed_data):
        """Combine results according to tensor_combinations configuration."""
        result_data = dict(data)
        
        # Constants for tensor dimensions
        frame_dim = 2
        channel_dim = 1
        
        # Create mask for conditioning
        mask = None
        if "vae" in processed_data and processed_data["vae"]:
            vae_results = list(processed_data["vae"].values())
            if vae_results:
                # Sort by position
                vae_results.sort(key=lambda x: x["position"])
                
                # Get example latent for shape
                example_latent = vae_results[0]["latents"]
                
                # Create mask for the length of all VAE frames
                mask = torch.zeros_like(example_latent[:, :1])  # Take only first channel
                mask_length = sum(r["frames"] for r in vae_results)
                mask[:, :, :mask_length] = 1
                
                # Process mask according to frame conditioning
                frame_cond_type = self.config.get("processors", {}).get("vae", {}).get("frame_conditioning", FrameConditioningType.FULL)
                frame_cond_index = self.config.get("processors", {}).get("vae", {}).get("frame_index", 0)
                
                # Match to expected video frames if available
                expected_frames = data["video"].shape[1] if "video" in data else None
                if expected_frames:
                    mask = apply_frame_conditioning_on_latents(
                        mask,
                        expected_frames,
                        channel_dim,
                        frame_dim,
                        frame_cond_type,
                        frame_cond_index,
                        False
                    )
                
                # Store the mask for tensor combinations
                processed_data["mask"] = {"mask": {"latents": mask, "position": 0}}
        
        # Get tensor combination configuration
        tensor_combinations = self.config.get("tensor_combinations", {})
        
        # Default combinations if not specified
        if not tensor_combinations:
            tensor_combinations = {
                "reference_latents": ["vae"],
                "mask_latents": ["mask"],
                "combined_condition_latents": ["vae", "mask"],
                "reference_embeddings": ["clip"]
            }
        
        # Process each defined output combination
        for output_name, processor_list in tensor_combinations.items():
            components = []
            
            # Collect tensors from each processor
            for proc_name in processor_list:
                if proc_name not in processed_data or not processed_data[proc_name]:
                    continue
                
                # Get all results for this processor
                proc_results = list(processed_data[proc_name].values())
                
                # Sort by position
                proc_results.sort(key=lambda x: x.get("position", 0))
                
                # Extract tensors based on processor type
                if proc_name == "vae":
                    # Concatenate VAE latents along frame dimension (dim=2)
                    if all("latents" in r for r in proc_results):
                        vae_latents = torch.cat([r["latents"] for r in proc_results], dim=frame_dim)
                        
                        # Apply frame conditioning to match expected video frames
                        expected_frames = data["video"].shape[1] if "video" in data else None
                        if expected_frames:
                            frame_cond_type = self.config.get("processors", {}).get("vae", {}).get("frame_conditioning", FrameConditioningType.FULL)
                            frame_cond_index = self.config.get("processors", {}).get("vae", {}).get("frame_index", 0)
                            
                            vae_latents = apply_frame_conditioning_on_latents(
                                vae_latents,
                                expected_frames,
                                channel_dim,
                                frame_dim,
                                frame_cond_type,
                                frame_cond_index,
                                False
                            )
                        
                        components.append(vae_latents)
                elif proc_name == "clip":
                    # For CLIP, extract features and concatenate along sequence dimension (dim=1)
                    clip_embeddings = [r for r in proc_results if r is not None]
                    if clip_embeddings:
                        clip_combined = torch.cat(clip_embeddings, dim=1)
                        components.append(clip_combined)
                elif proc_name == "mask":
                    # For mask tensors, just collect them directly
                    mask_tensors = [r["latents"] for r in proc_results if "latents" in r]
                    if mask_tensors:
                        components.extend(mask_tensors)
            
            # Skip if no components
            if not components:
                continue
            
            # Determine combine method based on output name
            if "latents" in output_name:
                # For latent outputs, concatenate along channel dimension (dim=1)
                result_data[f"e2v_{output_name}"] = torch.cat(components, dim=channel_dim)
            elif "embeddings" in output_name:
                # For embedding outputs, use as is (already concatenated above)
                result_data[f"e2v_{output_name}"] = components[0]
        
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