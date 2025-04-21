from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from .base import ProcessorMixin
import finetrainers.functional as FF
from finetrainers.logging import get_logger

logger = get_logger()


class BasePathwayProcessor(ProcessorMixin):
    """Base class for all pathway processors in E2V training."""
    
    def __init__(self, output_names=None, input_names=None, config=None, device=None):
        super().__init__()
        self.output_names = output_names or ["processor_output"]
        self.input_names = input_names or {}
        self.config = config or {}
        self.device = device
    
    def batch_process(self, model, inputs, configs, batch_size=None):
        """Process multiple inputs in batches.
        
        Args:
            model: The model to use for processing (VAE, CLIP, etc.)
            inputs: List of input tensors to process
            configs: List of configurations for each input
            batch_size: Maximum number of inputs to process in a single batch
            
        Returns:
            Dictionary mapping input indices to their processed results
        """
        if not inputs:
            return {}
            
        # Get batch size from config or use default
        if batch_size is None:
            batch_size = self.config.get("batch_size", 16)
            
        results = {}
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_configs = configs[i:i+batch_size]
            
            # Process batch
            batch_results = self._process_batch(model, batch_inputs, batch_configs)
            
            # Store results
            results.update(batch_results)
            
        return results
    
    def _process_batch(self, model, inputs, configs):
        """Process a batch of inputs.
        
        Args:
            model: The model to use for processing
            inputs: List of input tensors to process
            configs: List of configurations for each input
            
        Returns:
            Dictionary mapping input indices to their processed results
        """
        # Default implementation processes each input individually
        # Subclasses should override for more efficient batch processing
        results = {}
        
        for i, (input_data, config) in enumerate(zip(inputs, configs)):
            # Call the forward method directly for individual processing
            result = self.forward(model, image=input_data, element_config=config)
            results[i] = result[self.output_names[0]]
            
        return results


class VAEPathwayProcessor(BasePathwayProcessor):
    """Processor for the VAE spatial pathway in E2V training."""
    
    def forward(self, vae=None, image=None, video=None, element_config=None, **kwargs):
        """Process image/video through VAE pathway.
        
        Args:
            vae: VAE model for encoding
            image: Optional image tensor (B, C, H, W)
            video: Optional video tensor (B, F, C, H, W)
            element_config: Configuration for this element
            
        Returns:
            Dictionary with processed VAE output
        """
        # 1. Get configuration with element-specific overrides
        config = dict(self.config)
        if element_config and "processors" in element_config and "vae" in element_config["processors"]:
            config.update(element_config["processors"]["vae"])
        
        # 2. Preprocess image/video
        processed = self._preprocess_input(image, video, config)
        
        # 3. Apply repetition based on config
        repeated = self._apply_repetition(processed, config.get("repeat", 1))
        
        # 4. Encode through VAE if provided
        if vae is not None:
            encoded = self._encode_with_vae(repeated, vae)
        else:
            encoded = repeated
        
        # 5. Return result with metadata
        result = {
            "latents": encoded,
            "position": config.get("position", 0),
            "frames": encoded.shape[2] if len(encoded.shape) > 3 else 1
        }
        
        return {self.output_names[0]: result}
    
    def _preprocess_input(self, image, video, config):
        """Preprocess input image or video."""
        # Move to device if needed
        if image is not None and self.device is not None:
            image = image.to(self.device)
            
        if video is not None and self.device is not None:
            video = video.to(self.device)
        
        # Extract preprocessing parameters
        preprocessor = config.get("preprocessor", "letterbox")
        resolution = config.get("resolution", [480, 854])
        
        # Extract additional kwargs for preprocessor
        kwargs = {k: v for k, v in config.items() 
                  if k not in ["preprocessor", "resolution", "repeat", "position"]}
        
        if image is not None:
            # Apply preprocessing based on type
            if preprocessor == "letterbox":
                processed = FF.letterbox_image(image, resolution, **kwargs)
            elif preprocessor == "center_crop":
                processed = FF.center_crop_image(image, resolution)
            elif preprocessor == "resize_crop":
                processed = FF.resize_crop_image(image, resolution)
            else:
                logger.warning(f"Unknown preprocessor: {preprocessor}, using letterbox")
                processed = FF.letterbox_image(image, resolution, **kwargs)
                
            # For a single image, add a frame dimension if needed
            if len(processed.shape) == 4:  # (B, C, H, W)
                return processed.unsqueeze(2)  # (B, C, 1, H, W)
            if len(processed.shape) == 3:  # (C, H, W)
                return processed.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
            return processed
            
        elif video is not None:
            # For video, apply preprocessing to each frame
            # This is a simplified approach - full implementation would need frame-by-frame processing
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
            
    def _encode_with_vae(self, video, vae):
        """Encode video through VAE."""
        # Check if video has the right format
        if video.dim() != 5:  # (B, C, F, H, W)
            raise ValueError(f"Expected 5D tensor, got {video.dim()}D")
        
        # Move to the same device as the VAE
        device = vae.device
        video = video.to(device)
        
        # Encode through VAE
        with torch.no_grad():
            # For DiagonalGaussianDistribution output
            vae_output = vae.encode(video)
            if isinstance(vae_output, DiagonalGaussianDistribution):
                latents = vae_output.sample()
            else:
                latents = vae_output
            
            # Apply scaling if available in config
            if hasattr(vae, "config"):
                scale_factor = 1.0 / getattr(vae.config, "scaling_factor", 0.18215)
                latents = latents * scale_factor
        
        return latents
    
    def _process_batch(self, vae, inputs, configs):
        """Process a batch of inputs more efficiently."""
        # 1. Preprocess all inputs first
        processed_inputs = []
        for i, (input_data, config) in enumerate(zip(inputs, configs)):
            # Extract element-specific config for vae
            element_config = {}
            if "processors" in config and "vae" in config["processors"]:
                element_config = {"processors": {"vae": config["processors"]["vae"]}}
            
            # Preprocess
            processed = self._preprocess_input(input_data, None, dict(self.config, **element_config.get("processors", {}).get("vae", {})))
            
            # Apply repetition
            repeat = element_config.get("processors", {}).get("vae", {}).get("repeat", self.config.get("repeat", 1))
            repeated = self._apply_repetition(processed, repeat)
            
            processed_inputs.append(repeated)
        
        # 2. Try to stack inputs if possible (same shape)
        shapes = [p.shape for p in processed_inputs]
        if len(set(str(s) for s in shapes)) == 1:  # All shapes are the same
            # Stack inputs
            stacked_inputs = torch.cat(processed_inputs, dim=0)
            
            # 3. Encode through VAE in one go
            if vae is not None:
                encoded = self._encode_with_vae(stacked_inputs, vae)
                
                # 4. Split results
                results = {}
                for i, (_, config) in enumerate(zip(inputs, configs)):
                    element_config = {}
                    if "processors" in config and "vae" in config["processors"]:
                        element_config = {"processors": {"vae": config["processors"]["vae"]}}
                        
                    position = element_config.get("processors", {}).get("vae", {}).get("position", self.config.get("position", 0))
                    
                    result = {
                        "latents": encoded[i:i+1],
                        "position": position,
                        "frames": encoded.shape[2] if len(encoded.shape) > 3 else 1
                    }
                    results[i] = result
                    
                return results
        
        # Fallback to individual processing if shapes differ
        results = {}
        for i, (processed, config) in enumerate(zip(processed_inputs, configs)):
            element_config = {}
            if "processors" in config and "vae" in config["processors"]:
                element_config = {"processors": {"vae": config["processors"]["vae"]}}
                
            position = element_config.get("processors", {}).get("vae", {}).get("position", self.config.get("position", 0))
            
            if vae is not None:
                encoded = self._encode_with_vae(processed, vae)
            else:
                encoded = processed
                
            result = {
                "latents": encoded,
                "position": position,
                "frames": encoded.shape[2] if len(encoded.shape) > 3 else 1
            }
            results[i] = result
            
        return results


class CLIPPathwayProcessor(BasePathwayProcessor):
    """Processor for the CLIP semantic pathway in E2V training."""
    
    def forward(self, clip_vision_model=None, image=None, element_config=None, **kwargs):
        """Process image through CLIP pathway.
        
        Args:
            clip_vision_model: CLIP vision model for encoding
            image: Image tensor (B, C, H, W)
            element_config: Configuration for this element
            
        Returns:
            Dictionary with processed CLIP features
        """
        # 1. Get configuration with element-specific overrides
        config = dict(self.config)
        if element_config and "processors" in element_config and "clip" in element_config["processors"]:
            if isinstance(element_config["processors"]["clip"], dict):
                config.update(element_config["processors"]["clip"])
            elif not element_config["processors"]["clip"]:
                # CLIP pathway disabled for this element
                return {self.output_names[0]: None}
        
        # 2. Preprocess image
        processed = self._preprocess_input(image, config)
        
        # 3. Run CLIP encoder if available
        if clip_vision_model is not None:
            features = self._encode_with_clip(processed, clip_vision_model)
            
            # 4. Return result with metadata
            result = {
                "features": features,
                "position": config.get("position", 0)
            }
            return {self.output_names[0]: result}
        else:
            # If no CLIP model, just return preprocessed image
            result = {
                "preprocessed": processed,
                "position": config.get("position", 0)
            }
            return {self.output_names[0]: result}
    
    def _preprocess_input(self, image, config):
        """Preprocess image for CLIP."""
        if image is None:
            raise ValueError("Image must be provided for CLIP processing")
            
        # Move to device if needed
        if self.device is not None:
            image = image.to(self.device)
            
        # Extract preprocessing parameters
        preprocessor = config.get("preprocessor", "center_crop")
        resolution = config.get("resolution", [224, 224])
        
        # Extract additional kwargs for preprocessor
        kwargs = {k: v for k, v in config.items() 
                  if k not in ["preprocessor", "resolution", "position"]}
        
        # Apply preprocessing based on type
        if preprocessor == "center_crop":
            return FF.center_crop_image(image, resolution)
        elif preprocessor == "letterbox":
            return FF.letterbox_image(image, resolution, **kwargs)
        elif preprocessor == "resize_crop":
            return FF.resize_crop_image(image, resolution)
        else:
            logger.warning(f"Unknown preprocessor: {preprocessor}, using center_crop")
            return FF.center_crop_image(image, resolution)
    
    def _encode_with_clip(self, image, clip_vision_model):
        """Encode image with CLIP vision model."""
        # Move to the same device as the model
        device = clip_vision_model.device
        image = image.to(device)
        
        # Encode through CLIP vision model
        with torch.no_grad():
            # Get CLIP features
            outputs = clip_vision_model(image, output_hidden_states=True)
            
            # Extract features from the model's output
            if hasattr(outputs, "last_hidden_state"):
                # For transformers CLIPVisionModel
                features = outputs.last_hidden_state
            elif hasattr(outputs, "image_embeds"):
                # For open_clip models
                features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                # Another possible transformers format
                features = outputs.pooler_output.unsqueeze(1)  # Add sequence dim
            else:
                # Fallback - assume features are the direct output
                features = outputs
        
        return features
    
    def _process_batch(self, clip_vision_model, inputs, configs):
        """Process a batch of inputs more efficiently."""
        # 1. Preprocess all inputs
        processed_inputs = []
        positions = []
        for i, (input_data, config) in enumerate(zip(inputs, configs)):
            # Skip disabled elements
            if "processors" in config and "clip" in config["processors"] and not config["processors"]["clip"]:
                continue
                
            # Extract element-specific config for clip
            element_config = {}
            if "processors" in config and "clip" in config["processors"]:
                if isinstance(config["processors"]["clip"], dict):
                    element_config = {"processors": {"clip": config["processors"]["clip"]}}
            
            # Extract position
            position = element_config.get("processors", {}).get("clip", {}).get("position", self.config.get("position", 0))
            positions.append(position)
            
            # Preprocess
            processor_config = dict(self.config)
            if element_config and "processors" in element_config and "clip" in element_config["processors"]:
                processor_config.update(element_config["processors"]["clip"])
                
            processed = self._preprocess_input(input_data, processor_config)
            processed_inputs.append(processed)
        
        if not processed_inputs:
            return {}
        
        # 2. Try to stack inputs if possible (same shape)
        shapes = [p.shape for p in processed_inputs]
        if len(set(str(s) for s in shapes)) == 1:  # All shapes are the same
            # Stack inputs
            stacked_inputs = torch.cat(processed_inputs, dim=0)
            
            # 3. Encode through CLIP in one go
            if clip_vision_model is not None:
                features = self._encode_with_clip(stacked_inputs, clip_vision_model)
                
                # 4. Split results
                results = {}
                idx = 0
                for i, config in enumerate(configs):
                    # Skip disabled elements
                    if "processors" in config and "clip" in config["processors"] and not config["processors"]["clip"]:
                        results[i] = None
                        continue
                    
                    result = {
                        "features": features[idx:idx+1] if hasattr(features, "shape") else features,
                        "position": positions[idx]
                    }
                    results[i] = result
                    idx += 1
                    
                return results
            else:
                # No encoding, just preprocessed images
                results = {}
                idx = 0
                for i, config in enumerate(configs):
                    # Skip disabled elements
                    if "processors" in config and "clip" in config["processors"] and not config["processors"]["clip"]:
                        results[i] = None
                        continue
                    
                    result = {
                        "preprocessed": stacked_inputs[idx],
                        "position": positions[idx]
                    }
                    results[i] = result
                    idx += 1
                    
                return results
        
        # Fallback to individual processing if shapes differ
        results = {}
        idx = 0
        for i, (processed, config) in enumerate(zip(processed_inputs, configs)):
            # Skip disabled elements
            if "processors" in config and "clip" in config["processors"] and not config["processors"]["clip"]:
                results[i] = None
                continue
                
            if clip_vision_model is not None:
                features = self._encode_with_clip(processed, clip_vision_model)
                result = {
                    "features": features,
                    "position": positions[idx]
                }
            else:
                result = {
                    "preprocessed": processed,
                    "position": positions[idx]
                }
            
            results[i] = result
            idx += 1
            
        return results