"""Dataset components for E2V (Elements-to-Video) training.

This module handles loading and preprocessing elements for E2V training,
but does NOT perform any model inference to avoid CUDA multiprocessing issues.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.distributed.checkpoint.stateful
from PIL import Image
from diffusers.video_processor import VideoProcessor
from accelerate.data_loader import DataLoaderStateMixin

import finetrainers.functional as FF
from finetrainers.logging import get_logger

logger = get_logger()

class AccelerateDatasetStateWrapper(DataLoaderStateMixin):
    """Wrapper to provide Accelerate-compatible state fields."""
    
    def __init__(self):
        # Initialize Accelerate-specific state fields
        self.dl_state_dict = {
            "_sampler_iter_yielded": 0,
            "_sampler_indices_yielded": set(),
            "_indices_fetched_for_epoch": 0,
            "_prefetch_state": {}
        }

class IterableE2VDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    """Dataset wrapper for E2V training.
    
    This dataset wrapper:
    1. Identifies elements based on configured suffixes
    2. Preprocesses elements according to conditioning types
    3. Returns preprocessed data ready for model inference
    """
    
    def __init__(self, dataset, config, device=None):
        super().__init__()
        
        self.dataset = dataset
        self.config = config
        self.device = device
        
        # Extract configuration sections
        self.elements = config.get("elements", [])
        self.conditioning = config.get("conditioning", {})
        
        # Initialize video processor for preprocessing
        self.video_processor = VideoProcessor()
        
        # Create Accelerate state wrapper
        self._accelerate_state = AccelerateDatasetStateWrapper()
        
        logger.info(f"Initialized E2V dataset with {len(self.elements)} elements")
        for element in self.elements:
            logger.info(f"  Element: {element['name']}, suffixes: {element['suffixes']}")
    
    def __iter__(self):
        """Process dataset items according to configuration."""
        for data in iter(self.dataset):
            try:
                # 1. Identify elements from file paths
                element_files = self._find_element_files(data)
                
                # 2. Skip if required elements are missing
                if not self._check_required_elements(element_files):
                    continue
                
                # 3. Load and preprocess elements
                processed_data = self._preprocess_elements(data, element_files)
                
                # 4. Return data with preprocessed elements
                if processed_data:
                    result = {**data}
                    result["e2v_processed"] = processed_data
                    # Track yielded samples for Accelerate
                    self._accelerate_state.dl_state_dict["_sampler_iter_yielded"] += 1
                    yield result
                else:
                    logger.warning("No elements were successfully processed, skipping item")
            except Exception as e:
                logger.error(f"Error processing dataset item: {e}")
                continue
                
    def state_dict(self):
        """Return the state dictionary for checkpointing."""
        # Start with Accelerate state fields from wrapper
        state = self._accelerate_state.dl_state_dict.copy()
        
        # Add underlying dataset state if available
        if hasattr(self.dataset, "state_dict"):
            dataset_state = self.dataset.state_dict()
            # Ensure we don't overwrite Accelerate state fields if they exist in dataset state
            for k, v in dataset_state.items():
                if k not in state:
                    state[k] = v
            
        return state

    def load_state_dict(self, state_dict):
        """Load a state dictionary from a checkpoint."""
        # Load Accelerate state fields
        accelerate_fields = ["_sampler_iter_yielded", "_sampler_indices_yielded", 
                            "_indices_fetched_for_epoch", "_prefetch_state"]
        
        for field in accelerate_fields:
            if field in state_dict:
                self._accelerate_state.dl_state_dict[field] = state_dict[field]
            
        # Load underlying dataset state if available
        if hasattr(self.dataset, "load_state_dict"):
            # Create a dict with only the non-Accelerate fields
            dataset_state = {k: v for k, v in state_dict.items() 
                            if k not in accelerate_fields}
            if dataset_state:  # Only call if we have state to pass
                self.dataset.load_state_dict(dataset_state)
    
    
    def _find_element_files(self, data):
        """Match dataset files to configured elements based on suffixes."""
        element_files = {}
        
        # Process reference images if available
        if "images" in data and isinstance(data["images"], list):
            for image_path in data["images"]:
                filename = os.path.basename(image_path)
                
                # Match with configured elements
                for element in self.elements:
                    for suffix in element.get("suffixes", []):
                        if filename.endswith(suffix):
                            element_files[element["name"]] = {
                                "path": image_path,
                                "config": element
                            }
                            break
        
        # Add video data if available - could be provided as tensor directly
        if "video" in data:
            video_data = data["video"]
            
            # Find video element in configuration
            for element in self.elements:
                if element.get("name") == "video":
                    element_files["video"] = {
                        "tensor": video_data,
                        "config": element
                    }
                    break
        
        # Add caption/text if available
        if "caption" in data:
            caption = data["caption"]
            # Find caption element in configuration
            for element in self.elements:
                if element.get("name") == "captions":
                    element_files["captions"] = {
                        "text": caption,
                        "config": element
                    }
                    break
        
        return element_files
    
    def _check_required_elements(self, element_files):
        """Verify all required elements are present."""
        for element in self.elements:
            if element.get("required", False) and element["name"] not in element_files:
                logger.warning(f"Required element '{element['name']}' is missing")
                return False
        return True
    
    def _preprocess_elements(self, data, element_files):
        """Preprocess elements based on conditioning types."""
        # Create a shallow copy to store processed elements and add to original data
        processed = {}
        shallow_copy_data = dict(data.items())
        
        for element_name, file_info in element_files.items():
            element_config = file_info["config"]
            conditioning_type = element_config.get("conditioning")
            
            # Skip if no conditioning type specified
            if not conditioning_type or conditioning_type not in self.conditioning:
                continue
            
            conditioning_config = self.conditioning[conditioning_type]
            conditioning_processor = conditioning_config.get("type")
            
            # Process element based on conditioning type
            if conditioning_processor == "frame":
                self._process_frame_element(processed, element_name, file_info, conditioning_config)
            elif conditioning_processor == "clip":
                self._process_clip_element(processed, element_name, file_info, conditioning_config)
            elif conditioning_processor == "text":
                self._process_text_element(processed, element_name, file_info, conditioning_config)
            elif conditioning_processor == "video":
                self._process_video_element(processed, element_name, file_info, conditioning_config, shallow_copy_data)
            else:
                logger.warning(f"Unknown conditioning processor: {conditioning_processor}")
        
        return processed
    
    def _process_frame_element(self, processed, element_name, file_info, conditioning_config):
        """Process element for frame conditioning (VAE pathway)."""
        try:
            # Initialize frame processor section if needed
            if "frame" not in processed:
                processed["frame"] = {"elements": {}}
            
            element_config = file_info["config"]
            path = file_info.get("path")
            
            # Load image
            image = Image.open(path).convert("RGB")
            
            # Apply preprocessing based on configuration
            resolution = conditioning_config.get("resolution", [480, 854])
            preprocessor = conditioning_config.get("preprocessor", "letterbox")
            
            # Get element-specific VAE configuration
            vae_config = element_config.get("vae", {})
            position = vae_config.get("position", 0)
            repeat = vae_config.get("repeat", 1)
            
            # Process image through appropriate preprocessor
            if preprocessor == "letterbox":
                tensor = FF.letterbox_image(
                    self.video_processor.preprocess(image), 
                    resolution
                )
            elif preprocessor == "center_crop":
                tensor = FF.center_crop_image(
                    self.video_processor.preprocess(image),
                    resolution
                )
            elif preprocessor == "resize":
                tensor = FF.resize_image(
                    self.video_processor.preprocess(image),
                    resolution
                )
            else:
                # Default to letterbox
                tensor = FF.letterbox_image(
                    self.video_processor.preprocess(image),
                    resolution
                )
            
            # Add frame dimension if needed (B, C, H, W) -> (B, C, 1, H, W)
            if len(tensor.shape) == 4:
                tensor = tensor.unsqueeze(2)
            
            # Store processed tensor and metadata
            processed["frame"]["elements"][element_name] = {
                "tensor": tensor,
                "position": position,
                "repeat": repeat
            }
            
            # Store global frame conditioning parameters
            processed["frame"]["conditioning"] = {
                "frame_conditioning_type": conditioning_config.get("frame_conditioning_type", "full"),
                "frame_conditioning_concatenate_mask": conditioning_config.get("frame_conditioning_concatenate_mask", True),
                "frame_conditioning_index": conditioning_config.get("frame_conditioning_index", 0)
            }
            
        except Exception as e:
            logger.error(f"Error processing frame element {element_name}: {e}")
    
    def _process_clip_element(self, processed, element_name, file_info, conditioning_config):
        """Process element for CLIP conditioning (semantic pathway)."""
        try:
            # Initialize clip processor section if needed
            if "clip" not in processed:
                processed["clip"] = {"elements": {}}
            
            element_config = file_info["config"]
            path = file_info.get("path")
            
            # Load image
            image = Image.open(path).convert("RGB")
            
            # Apply preprocessing based on configuration
            resolution = conditioning_config.get("resolution", [224, 224])
            preprocessor = conditioning_config.get("preprocessor", "center_crop")
            
            # Get element-specific CLIP configuration
            clip_config = element_config.get("clip", {})
            position = clip_config.get("position", 0)
            
            # Process image through appropriate preprocessor
            if preprocessor == "center_crop":
                tensor = FF.center_crop_image(
                    self.video_processor.preprocess(image),
                    resolution
                )
            elif preprocessor == "letterbox":
                tensor = FF.letterbox_image(
                    self.video_processor.preprocess(image),
                    resolution
                )
            elif preprocessor == "resize":
                tensor = FF.resize_image(
                    self.video_processor.preprocess(image),
                    resolution
                )
            else:
                # Default to center crop
                tensor = FF.center_crop_image(
                    self.video_processor.preprocess(image),
                    resolution
                )
            
            # Store processed tensor and metadata
            processed["clip"]["elements"][element_name] = {
                "tensor": tensor,
                "position": position
            }
            
        except Exception as e:
            logger.error(f"Error processing CLIP element {element_name}: {e}")
    
    def _process_text_element(self, processed, element_name, file_info, conditioning_config):
        """Process element for text conditioning."""
        try:
            # Initialize text processor section if needed
            if "text" not in processed:
                processed["text"] = {"elements": {}}
            
            element_config = file_info["config"]
            
            # Get text from file or data
            text = file_info.get("text")
            if not text and "path" in file_info:
                # Read from file if path is provided
                with open(file_info["path"], "r") as f:
                    text = f.read().strip()
            
            # Apply preprocessing if needed
            if conditioning_config.get("remove_common_llm_caption_prefixes", False):
                # Simple prefix removal - more complex in real implementation
                common_prefixes = ["A picture of ", "An image of "]
                for prefix in common_prefixes:
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                        break
            
            # Store processed text and metadata
            processed["text"]["elements"][element_name] = {
                "text": text
            }
            
        except Exception as e:
            logger.error(f"Error processing text element {element_name}: {e}")
    
    def _process_video_element(self, processed, element_name, file_info, conditioning_config, data_dict):
        """Process element for video conditioning (target video)."""
        try:
            # Import necessary modules
            import torch
            import finetrainers.functional as FF
            
            # Initialize video processor section if needed
            if "video" not in processed:
                processed["video"] = {"elements": {}}
            
            element_config = file_info["config"]
            path = file_info.get("path")
            tensor = file_info.get("tensor")
            
            # If we have a tensor directly, use it
            if tensor is not None and isinstance(tensor, torch.Tensor):
                video_tensor = tensor
            elif path is not None:
                # Load video from path (not needed for E2V implementation)
                # This is just a placeholder for future extensions
                logger.warning(f"Video loading from path not implemented: {path}")
                return
            else:
                logger.warning(f"No video tensor or path provided for element {element_name}")
                return
            
            # Apply preprocessing based on configuration
            resolution = conditioning_config.get("resolution", [480, 854])
            preprocessor = conditioning_config.get("preprocessor", "bicubic")
            
            # Process video through appropriate preprocessor
            if preprocessor == "bicubic":
                # For 4D tensor [B, C, H, W]
                if len(video_tensor.shape) == 4:
                    processed_tensor = FF.resize_image(video_tensor, resolution, mode="bicubic")
                # For 5D tensor [B, C, F, H, W]
                elif len(video_tensor.shape) == 5:
                    batch_size, channels, frames, height, width = video_tensor.shape
                    reshaped = video_tensor.reshape(-1, channels, height, width)
                    resized = FF.resize_image(reshaped, resolution, mode="bicubic")
                    processed_tensor = resized.reshape(batch_size, channels, frames, *resolution)
                else:
                    logger.warning(f"Unexpected tensor shape for video: {video_tensor.shape}")
                    return
            elif preprocessor == "center_crop":
                if len(video_tensor.shape) == 5:  # [B, C, F, H, W]
                    processed_tensor = FF.center_crop_video(video_tensor, resolution)
                else:
                    processed_tensor = FF.center_crop_image(video_tensor, resolution)
            else:
                # Default to bicubic resize
                if len(video_tensor.shape) == 4:
                    processed_tensor = FF.resize_image(video_tensor, resolution, mode="bicubic")
                elif len(video_tensor.shape) == 5:
                    batch_size, channels, frames, height, width = video_tensor.shape
                    reshaped = video_tensor.reshape(-1, channels, height, width)
                    resized = FF.resize_image(reshaped, resolution, mode="bicubic")
                    processed_tensor = resized.reshape(batch_size, channels, frames, *resolution)
            
            # Store the processed tensor in the processed data
            processed["video"]["elements"][element_name] = {
                "tensor": processed_tensor
            }
            
            # Also update the data dictionary with the processed video tensor
            # This makes it available for the trainer to use directly
            data_dict["video"] = processed_tensor
            
        except Exception as e:
            logger.error(f"Error processing video element {element_name}: {e}")


class ValidationE2VDataset(IterableE2VDataset):
    """Validation dataset for E2V training.
    
    Extends IterableE2VDataset with validation-specific functionality.
    """
    
    def __init__(self, dataset, config, device=None):
        super().__init__(dataset, config, device)
        # The parent class already initializes the Accelerate state wrapper
    
    def __iter__(self):
        """Process dataset items for validation."""
        for data in super().__iter__():
            # For validation we want to include original elements
            # for visualization purposes
            if "e2v_processed" in data:
                data["element_files"] = {k: v.get("path", v.get("text", "")) 
                                         for k, v in data.get("e2v_elements", {}).items()}
            # Parent already increments the state counters
            yield data
