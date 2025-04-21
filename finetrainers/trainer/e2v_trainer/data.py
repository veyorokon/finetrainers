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

from .config import ElementConfig, FrameConditioningType


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
    logger = get_logger()
    num_frames = latents.size(frame_dim)
    logger.info(f"Frame conditioning - input: shape={latents.shape}, expected_frames={expected_num_frames}")
    
    mask = torch.zeros_like(latents)
    logger.info(f"Created mask with shape={mask.shape}")

    if frame_conditioning_type == FrameConditioningType.INDEX:
        frame_index = min(frame_conditioning_index or 0, num_frames - 1)
        logger.info(f"Using INDEX conditioning with frame_index={frame_index}")
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = frame_index
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.PREFIX:
        frame_index = random.randint(1, num_frames)
        logger.info(f"Using PREFIX conditioning with random frame_index={frame_index}")
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = slice(0, frame_index)  # Keep frames 0 to frame_index-1
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.RANDOM:
        # Zero or more random frames to keep
        num_frames_to_keep = random.randint(1, num_frames)
        frame_indices = random.sample(range(num_frames), num_frames_to_keep)
        logger.info(f"Using RANDOM conditioning, keeping {num_frames_to_keep} frames: {frame_indices}")
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = frame_indices
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.FIRST_AND_LAST:
        logger.info(f"Using FIRST_AND_LAST conditioning")
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = 0
        mask[tuple(indexing)] = 1
        indexing[frame_dim] = num_frames - 1
        mask[tuple(indexing)] = 1
        latents = latents * mask

    elif frame_conditioning_type == FrameConditioningType.FULL:
        logger.info(f"Using FULL conditioning (all frames)")
        indexing = [slice(None)] * latents.ndim
        indexing[frame_dim] = slice(0, num_frames)
        mask[tuple(indexing)] = 1
        
    # Handle padding/truncation to match expected number of frames
    if latents.size(frame_dim) >= expected_num_frames:
        logger.info(f"Truncating from {latents.size(frame_dim)} to {expected_num_frames} frames")
        slicing = [slice(None)] * latents.ndim
        slicing[frame_dim] = slice(expected_num_frames)
        latents = latents[tuple(slicing)]
        mask = mask[tuple(slicing)]
    else:
        logger.info(f"Padding from {latents.size(frame_dim)} to {expected_num_frames} frames")
        pad_size = expected_num_frames - num_frames
        pad_shape = list(latents.shape)
        pad_shape[frame_dim] = pad_size
        padding = latents.new_zeros(pad_shape)
        logger.info(f"Created padding with shape={padding.shape}")
        latents = torch.cat([latents, padding], dim=frame_dim)
        mask = torch.cat([mask, padding], dim=frame_dim)

    logger.info(f"After padding/truncation: latents={latents.shape}, mask={mask.shape}")
    
    if concatenate_mask:
        logger.info(f"Concatenating mask along channel dim {channel_dim}")
        try:
            latents = torch.cat([latents, mask], dim=channel_dim)
            logger.info(f"After concatenating mask: shape={latents.shape}")
        except RuntimeError as e:
            logger.error(f"Error concatenating mask: {e}")
            logger.error(f"latents shape: {latents.shape}, mask shape: {mask.shape}")
            raise

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
        
        
        # Initialize processors based on configuration
        self.processors = {}
        
        # Log processor initialization
        logger.info(f"Initializing E2V processors. Have CLIP processor: {clip_processor is not None}")
        
        # Initialize all processors defined in the configuration
        for proc_name, proc_config in config.get("processors", {}).items():
            if proc_name == "vae":
                self.processors["vae"] = VAEPathwayProcessor(
                    output_names=["vae_output"],
                    config=proc_config,
                    device=device
                )
                logger.info(f"Initialized VAE processor with config: {proc_config}")
            elif proc_name == "clip":
                if clip_processor is None:
                    logger.warning(f"CLIP processor requested but no clip_processor provided - CLIP pathway will not work")
                    # Skip initializing CLIP processor if no model available
                    if "tensor_combinations" in config:
                        # Check if CLIP is required in tensor_combinations
                        is_clip_required = any("clip" in procs for procs in config["tensor_combinations"].values())
                        if is_clip_required:
                            logger.warning(f"CLIP is required in tensor_combinations but no clip_processor provided")
                            logger.warning(f"Updating tensor_combinations to remove CLIP dependency as a temporary fix")
                            # Modify the tensor_combinations to remove clip dependency
                            for key, procs in list(config["tensor_combinations"].items()):
                                config["tensor_combinations"][key] = [p for p in procs if p != "clip"]
                else:
                    self.processors["clip"] = CLIPPathwayProcessor(
                        output_names=["clip_output"],
                        config=proc_config,
                        device=device,
                        clip_processor=clip_processor
                    )
                    logger.info(f"Initialized CLIP processor with config: {proc_config}")
            else:
                logger.warning(f"Unknown processor type: {proc_name}")
        
        logger.info(f"Initialized processors: {list(self.processors.keys())}")
        
        # Check if required processors for tensor_combinations are available
        if "tensor_combinations" in config:
            all_procs = set(sum(config["tensor_combinations"].values(), []))
            missing_procs = [p for p in all_procs if p not in self.processors]
            if missing_procs:
                logger.warning(f"Missing required processors for tensor_combinations: {missing_procs}")
                logger.warning(f"This may cause errors during processing")
        
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
        
        
        # Map real elements to processors
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
        
        # Get tensor combination configuration - must be explicitly specified
        tensor_combinations = self.config.get("tensor_combinations", {})
        
        # First collect all processor outputs by type
        collected_tensors = {}
        
        # Helper function to extract and process tensors from a processor
        def collect_processor_tensors(proc_name):
            if proc_name not in processed_data or not processed_data[proc_name]:
                logger.error(f"Required processor {proc_name} not found in processed data")
                raise ValueError(f"Required processor {proc_name} not found in processed data")
            
            # Get all results for this processor type
            proc_results = list(processed_data[proc_name].values())
            if not proc_results:
                logger.error(f"No results from processor {proc_name}")
                raise ValueError(f"No results from processor {proc_name}")
                
            # Sort by position
            proc_results.sort(key=lambda x: x.get("position", 0))
            
            # Different handling based on processor type
            field_name = "latents"  # Default field name
            concat_dim = frame_dim  # Default concatenation dimension
            
            # For CLIP processor, we concatenate along sequence dimension
            if proc_name == "clip":
                concat_dim = 1  # Sequence dimension for embeddings
            
            # Extract tensors from the results
            tensors = [r.get(field_name) for r in proc_results if field_name in r]
            if not tensors:
                logger.error(f"No {field_name} found in results for processor {proc_name}")
                raise ValueError(f"No {field_name} found in results for processor {proc_name}")
                
            # Log available fields in processor results for debugging
            for i, r in enumerate(proc_results):
                logger.info(f"  Result {i} from {proc_name} has fields: {list(r.keys())}")
            
            # Log tensor shapes before concatenation
            logger.info(f"Tensors for {proc_name} before concatenation:")
            for i, t in enumerate(tensors):
                logger.info(f"  Tensor {i}: shape={t.shape}")
            
            # Concatenate tensors
            try:
                combined = torch.cat(tensors, dim=concat_dim)
                logger.info(f"Successfully concatenated {len(tensors)} tensors for {proc_name}: result shape={combined.shape}")
            except RuntimeError as e:
                logger.error(f"Failed to concatenate tensors for {proc_name}: {e}")
                logger.error(f"Tensor shapes: {[t.shape for t in tensors]}")
                logger.error(f"Concatenation dimension: {concat_dim}")
                raise
            
            # Apply frame conditioning for tensors with frames
            if proc_name == "vae" and "video" in data:
                expected_frames = data["video"].shape[1]
                if expected_frames and len(combined.shape) >= 5:  # Has frame dimension
                    processor_config = self.config.get("processors", {}).get(proc_name, {})
                    frame_cond_type = processor_config.get("frame_conditioning", FrameConditioningType.FULL)
                    frame_cond_index = processor_config.get("frame_index", 0)
                    concatenate_mask = processor_config.get("concatenate_mask", True)
                    
                    logger.info(f"Applying frame conditioning: type={frame_cond_type}, index={frame_cond_index}, concat_mask={concatenate_mask}")
                    logger.info(f"Combined shape before frame conditioning: {combined.shape}")
                    logger.info(f"Video shape: {data['video'].shape}, expected_frames={expected_frames}")
                    
                    try:
                        combined = apply_frame_conditioning_on_latents(
                            combined,
                            expected_frames,
                            channel_dim,
                            frame_dim,
                            frame_cond_type,
                            frame_cond_index,
                            concatenate_mask
                        )
                        logger.info(f"Combined shape after frame conditioning: {combined.shape}")
                    except Exception as e:
                        logger.error(f"Error during frame conditioning: {e}")
                        # Continue without frame conditioning
                        logger.warning("Skipping frame conditioning due to error")
            
            return combined
        
        # Log tensor combinations for debugging
        logger.info(f"Using tensor_combinations: {tensor_combinations}")
        
        # Get list of all required processors from tensor_combinations
        required_procs = set(sum(tensor_combinations.values(), []))
        logger.info(f"Required processors from tensor_combinations: {required_procs}")
        logger.info(f"Available processors: {list(self.processors.keys())}")
        
        # Process all processor types
        for proc_name in required_procs:
            if proc_name not in self.processors:
                logger.error(f"Required processor '{proc_name}' not found in available processors {list(self.processors.keys())}")
                raise ValueError(f"Required processor '{proc_name}' not available - check tensor_combinations configuration")
                
            # Process this processor type
            try:
                collected_tensors[proc_name] = collect_processor_tensors(proc_name)
                # Log the shape of each collected tensor
                logger.info(f"Collected tensor for {proc_name}: shape={collected_tensors[proc_name].shape}")
            except Exception as e:
                logger.error(f"Error collecting tensors for processor {proc_name}: {e}")
                raise
        
        # Now create the output combinations
        for output_name, processor_list in tensor_combinations.items():
            components = []
            logger.info(f"Creating output combination '{output_name}' from processors: {processor_list}")
            
            # Collect tensors from each processor in the list
            for proc_name in processor_list:
                if proc_name not in collected_tensors:
                    logger.error(f"Required processor {proc_name} not found in collected tensors")
                    raise ValueError(f"Required processor {proc_name} not found in collected tensors")
                
                tensor = collected_tensors[proc_name]
                logger.info(f"Adding component for {output_name} from {proc_name}: shape={tensor.shape}")
                components.append(tensor)
            
            # Output combinations must have at least one component
            if not components:
                logger.error(f"No components available for output {output_name}")
                raise ValueError(f"No components available for output {output_name}")
            
            # Determine combine method based on number of components
            if len(components) == 1:
                # Single component, no need to concatenate
                result_data[f"e2v_{output_name}"] = components[0]
                logger.info(f"Created {output_name} from single component: shape={components[0].shape}")
            else:
                # Multiple components, concatenate along channel dimension
                logger.info(f"Attempting to concatenate for {output_name}: {[c.shape for c in components]}")
                try:
                    result_data[f"e2v_{output_name}"] = torch.cat(components, dim=channel_dim)
                    logger.info(f"Created {output_name} by concatenating tensors: final shape={result_data[f'e2v_{output_name}'].shape}")
                except RuntimeError as e:
                    logger.error(f"Failed to concatenate tensors for {output_name}: {e}")
                    logger.error(f"Tensor shapes: {[c.shape for c in components]}")
                    logger.error(f"Target concatenation dimension: {channel_dim}")
                    raise
        
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