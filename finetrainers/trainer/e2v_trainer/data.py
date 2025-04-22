"""Dataset components for E2V (Elements-to-Video) training.

This module handles loading and preprocessing elements for E2V training,
but does NOT perform any model inference to avoid CUDA multiprocessing issues.

Model inference is handled in the trainer to keep all CUDA operations in
the main process.
"""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed.checkpoint.stateful
from PIL import Image
from diffusers.video_processor import VideoProcessor

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.processors import ProcessorMixin
from finetrainers.typing import ArtifactType

from .config import ElementConfig, FrameConditioningType
from .utils import is_processor_enabled, get_processor_config, validate_e2v_config

logger = get_logger()


class IterableE2VDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    """Dataset wrapper for E2V (Elements-to-Video) training.
    
    This wrapper handles loading and preprocessing elements for E2V training:
    1. Identifies and loads reference images for each video
    2. Preprocesses images (resize, crop, etc.) but does NOT run model inference
    3. Returns preprocessed data for trainer to process through models
    
    Model inference (VAE encoding, CLIP processing) is handled by the trainer
    to avoid CUDA multiprocessing issues.
    """
    
    def __init__(self, dataset, config, device=None):
        super().__init__()
        
        self.dataset = dataset
        self.config = config
        self.device = device
        
        # Validate configuration
        try:
            validate_e2v_config(config)
        except ValueError as e:
            raise ValueError(f"Invalid E2V configuration: {e}")
        
        # Get configuration sections
        self.elements = config.get("elements", [])
        self.processors_config = config.get("processors", {})
        self.tensor_combinations = config.get("tensor_combinations", {})
        
        logger.info(f"Initialized IterableE2VDataset with {len(self.elements)} elements")
        logger.info(f"Using processors: {list(self.processors_config.keys())}")
        logger.info(f"Using tensor combinations: {self.tensor_combinations}")
    
    def __iter__(self):
        """Iterate through dataset and yield preprocessed elements.
        
        This method:
        1. Loads raw element images
        2. Preprocesses them (resize, crop, etc.)
        3. Returns preprocessed tensors WITHOUT model inference
        
        Model inference happens in the trainer to avoid CUDA multiprocessing issues.
        """
        for data in iter(self.dataset):
            try:
                # Find element files based on dataset item
                element_files = self._find_element_files(data)
                
                # Skip items where required elements are missing
                missing_required = False
                for element in self.elements:
                    if element.get("required", False) and element["name"] not in element_files:
                        logger.warning(f"Required element '{element['name']}' missing, skipping item")
                        missing_required = True
                        break
                
                if missing_required:
                    continue
                
                # Load element images
                element_data = self._load_elements(element_files)
                
                # If loading failed, skip this item
                if not element_data:
                    logger.warning("No elements could be loaded, skipping item")
                    continue
                
                # Preprocess elements (resize, crop, etc.) but don't run models
                preprocessed_data = self._preprocess_elements(data, element_data)
                
                # Create output dictionary with preprocessed data
                result_data = dict(data)
                result_data["preprocessed_elements"] = preprocessed_data
                result_data["element_configs"] = {}
                
                # Include element configs for the trainer
                for name, element in element_data.items():
                    if "config" in element:
                        result_data["element_configs"][name] = element["config"]
                
                # Include processor configurations
                result_data["processor_configs"] = self.processors_config
                
                # Include tensor combinations
                result_data["tensor_combinations"] = self.tensor_combinations
                
                yield result_data
            except Exception as e:
                logger.error(f"Error preprocessing dataset item: {e}")
                # Skip this item and continue
                continue
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.dataset.load_state_dict(state_dict)
    
    def state_dict(self):
        """Save state for checkpoint."""
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
        
        # Debug what dataset provides
        logger.debug(f"Dataset item keys: {list(data.keys())}")
        
        # Check if we have reference images from our VideoReferenceImagesDataset
        if "images" in data:
            logger.debug(f"Found {len(data['images'])} reference images in dataset item")
            
            # Process each reference image
            for image_path in data["images"]:
                # Get the basename to match with suffixes
                filename = os.path.basename(image_path)
                
                # Try to match with one of our configured element types
                for element_config in self.elements:
                    # Check if this image matches one of the element's suffixes
                    for config_suffix in element_config.get("suffixes", []):
                        if filename.endswith(config_suffix):
                            element_files[element_config["name"]] = {
                                "path": image_path,
                                "config": element_config
                            }
                            logger.debug(f"Found {element_config['name']} reference image: {image_path}")
                            break
        else:
            logger.warning("No reference images ('images' key) found in dataset item")
        
        return element_files
    
    def _load_elements(self, element_files):
        """Load element images from files.
        
        Args:
            element_files: Dictionary mapping element names to file info
            
        Returns:
            Dictionary mapping element names to loaded image data
        """
        element_data = {}
        
        # Load each element
        for element_name, file_info in element_files.items():
            try:
                # Load image
                image_path = file_info["path"]
                
                # Load and process image
                element_img = Image.open(image_path).convert("RGB")
                
                # Convert to tensor using VideoProcessor
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
    
    def _preprocess_elements(self, data, element_data):
        """Preprocess elements for each pathway (resize, crop, etc.).
        
        This method ONLY handles preprocessing, not model inference:
        1. Resizes and crops images based on config
        2. Returns tensors ready for model inference in the trainer
        
        No VAE encoding or CLIP processing is done here - that happens in the trainer.
        
        Args:
            data: Original dataset item
            element_data: Dictionary mapping element names to loaded image data
            
        Returns:
            Dictionary mapping processor names to preprocessed elements
        """
        preprocessed = {}
        
        # Process for each configured processor type
        for proc_name, proc_config in self.processors_config.items():
            # Initialize processor section in result
            preprocessed[proc_name] = {}
            
            # Process each element
            for element_name, element_info in element_data.items():
                element_img = element_info["image"]
                element_config = element_info["config"]
                
                # Skip if processor is disabled for this element
                if not is_processor_enabled(element_config, proc_name):
                    logger.debug(f"Processor {proc_name} disabled for element {element_name}, skipping")
                    continue
                
                # Get processor-specific config with element-specific overrides
                merged_config = get_processor_config(element_config, proc_name, proc_config)
                
                # Apply preprocessing based on processor type
                if proc_name == "vae":
                    # Default to letterbox preprocessing for VAE
                    preprocessor = merged_config.get("preprocessor", "letterbox")
                    resolution = merged_config.get("resolution", [480, 854])
                    
                    if preprocessor == "letterbox":
                        processed = FF.letterbox_image(element_img, resolution)
                    elif preprocessor == "center_crop":
                        processed = FF.center_crop_image(element_img, resolution)
                    elif preprocessor == "resize":
                        processed = FF.resize_image(element_img, resolution)
                    else:
                        # Default to letterbox
                        processed = FF.letterbox_image(element_img, resolution)
                        
                elif proc_name == "clip":
                    # Default to center_crop preprocessing for CLIP
                    preprocessor = merged_config.get("preprocessor", "center_crop")
                    resolution = merged_config.get("resolution", [224, 224])
                    
                    if preprocessor == "letterbox":
                        processed = FF.letterbox_image(element_img, resolution)
                    elif preprocessor == "center_crop":
                        processed = FF.center_crop_image(element_img, resolution)
                    elif preprocessor == "resize":
                        processed = FF.resize_image(element_img, resolution)
                    else:
                        # Default to center_crop
                        processed = FF.center_crop_image(element_img, resolution)
                        
                else:
                    # Unknown processor type, skip
                    logger.warning(f"Unknown processor type: {proc_name}, skipping")
                    continue
                
                # Store preprocessed tensor with metadata
                preprocessed[proc_name][element_name] = {
                    "tensor": processed,
                    "position": merged_config.get("position", 0),
                    "repeat": merged_config.get("repeat", 1),
                    "config": merged_config
                }
        
        return preprocessed


class ValidationE2VDataset(IterableE2VDataset):
    """Validation dataset for E2V training.
    
    Same as IterableE2VDataset but also includes original element files
    for visualization during validation.
    """
    
    def __iter__(self):
        """Iterate through validation dataset.
        
        Same as IterableE2VDataset.__iter__ but also includes
        original element files for visualization.
        """
        for data in super().__iter__():
            # Include original element files for visualization
            if "element_files" not in data and hasattr(self, "_find_element_files"):
                data["element_files"] = self._find_element_files(data)
            
            yield data
    
    # Inherit state_dict method from parent class
