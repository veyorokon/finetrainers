"""Encoder functions for E2V processor types.

This module contains registered encoder functions for processing
elements through various models (VAE, CLIP, etc.)
"""
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torchvision import transforms

from finetrainers.logging import get_logger
from finetrainers.processors import ProcessorMixin

logger = get_logger()

# Registry to store encoder functions
ENCODER_REGISTRY = {}


def register_encoder(name):
    """Register an encoder function for a processor type."""
    def decorator(func):
        ENCODER_REGISTRY[name] = func
        return func
    return decorator


@register_encoder("vae")
def encode_vae(tensor, model, config=None):
    """Encode tensor with VAE model.
    
    Args:
        tensor: Tensor to encode
        model: VAE model
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary with encoded latents and metadata
    """
    config = config or {}
    
    # Add detailed logging
    logger.debug(f"VAE encoding tensor with shape: {tensor.shape}, dtype: {tensor.dtype}")
    logger.debug(f"VAE config: {config}")
    
    try:
        # Move to model device and match model dtype
        device = model.device
        
        # Get model dtype from model parameters
        model_dtype = next(model.parameters()).dtype
        logger.debug(f"Moving tensor to device {device} with dtype {model_dtype}")
        
        # Convert tensor to match model dtype
        tensor = tensor.to(device, dtype=model_dtype)
        
        # Check if tensor is 2D, 3D, 4D, or 5D (B, C, [F], H, W)
        tensor_dim = len(tensor.shape)
        logger.debug(f"Input tensor dimension: {tensor_dim}")
        
        # Handle dimension issues - ensure we have a 5D tensor for video or 4D for image
        if tensor_dim == 2:  # Likely [H, W]
            logger.debug("Converting 2D tensor to 4D [1, 3, H, W]")
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            if tensor.shape[1] == 1:  # Add channels if needed
                tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor_dim == 3:  # Likely [C, H, W]
            logger.debug("Converting 3D tensor to 4D [1, C, H, W]")
            tensor = tensor.unsqueeze(0)
        elif tensor_dim == 4:  # [B, C, H, W] - image format, add time dimension for video
            logger.debug("Input is 4D (image format)")
            # For video VAE, we need 5D tensor [B, C, F, H, W]
            if hasattr(model, "is_vae_video") and model.is_vae_video:
                logger.debug("Adding time dimension for video VAE")
                tensor = tensor.unsqueeze(2)  # Add frame dimension
        
        logger.debug(f"Tensor shape after dimension adjustment: {tensor.shape}")
        
        # Apply repetition if specified
        repeat = config.get("repeat", 1)
        if repeat > 1:
            if len(tensor.shape) >= 5:  # B, C, F, H, W (video)
                # Determine which frames to repeat
                frame_dim = 2
                frames = tensor.shape[frame_dim]
                
                logger.debug(f"Repeating {frames} frames {repeat} times each")
                
                # Handle single frame case
                if frames == 1:
                    tensor = torch.cat([tensor] * repeat, dim=frame_dim)
                else:
                    # For multiple frames, repeat each frame as specified
                    repeated_frames = []
                    for i in range(frames):
                        frame = tensor[:, :, i:i+1, :, :]
                        repeated_frames.append(torch.cat([frame] * repeat, dim=frame_dim))
                    
                    tensor = torch.cat(repeated_frames, dim=frame_dim)
            else:  # Handle image case (4D tensor)
                logger.debug(f"Tensor is 4D, creating time dimension before repeating")
                # Add time dimension, then repeat
                tensor = tensor.unsqueeze(2)  # [B, C, 1, H, W]
                tensor = torch.cat([tensor] * repeat, dim=2)  # [B, C, repeat, H, W]
        
        logger.debug(f"Tensor shape after repetition: {tensor.shape}")
        
        # Encode through VAE
        with torch.no_grad():
            # Encode tensor through VAE
            logger.debug(f"Sending to VAE encoder: shape={tensor.shape}")
            vae_output = model.encode(tensor)
            
            # Handle DiagonalGaussianDistribution output
            if isinstance(vae_output, DiagonalGaussianDistribution):
                latents = vae_output.sample()
                logger.debug(f"Sampled from DiagonalGaussianDistribution: shape={latents.shape}")
            else:
                latents = vae_output
                logger.debug(f"Used VAE output directly: shape={latents.shape}")
            
            # Apply VAE scaling
            scale_factor = 1.0 / getattr(model.config, "scaling_factor", 0.18215)
            latents = latents * scale_factor
        
        # Return latents and metadata
        result = {
            "latents": latents,
            "position": config.get("position", 0),
            "frames": latents.shape[2] if len(latents.shape) > 3 else 1
        }
        logger.debug(f"VAE encoding complete: latent shape={latents.shape}, frames={result['frames']}")
        return result
        
    except Exception as e:
        logger.error(f"VAE encoding error: {e}", exc_info=True)
        logger.error(f"Tensor shape: {tensor.shape if 'tensor' in locals() else 'unknown'}")
        logger.error(f"VAE model type: {type(model)}")
        logger.error(f"Config: {config}")
        # Re-raise for proper error handling
        raise


@register_encoder("clip")
def encode_clip(tensor, model, config=None):
    """Encode tensor with CLIP model.
    
    Args:
        tensor: Tensor to encode
        model: CLIP model
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary with encoded features and metadata
    """
    config = config or {}
    
    # Add detailed logging
    logger.debug(f"CLIP encoding tensor with shape: {tensor.shape}, dtype: {tensor.dtype}")
    logger.debug(f"CLIP config: {config}")
    
    try:
        # Move to model device and match model dtype
        device = model.device
        
        # Get model dtype from model parameters
        model_dtype = next(model.parameters()).dtype
        logger.debug(f"Moving tensor to device {device} with dtype {model_dtype}")
        
        # Convert tensor to match model dtype
        tensor = tensor.to(device, dtype=model_dtype)
        
        # Check tensor dimensions
        tensor_dim = len(tensor.shape)
        logger.debug(f"Input tensor dimension: {tensor_dim}")
        
        # Handle dimension issues - CLIP expects [B, C, H, W]
        if tensor_dim == 2:  # [H, W]
            logger.debug("Converting 2D tensor to 4D [1, 3, H, W]")
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            if tensor.shape[1] == 1:  # Add channels
                tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor_dim == 3:  # [C, H, W]
            logger.debug("Converting 3D tensor to 4D [1, C, H, W]")
            tensor = tensor.unsqueeze(0)
        elif tensor_dim == 5:  # [B, C, F, H, W] - video format
            logger.debug("Converting 5D tensor to 4D by taking first frame")
            # Take first frame for CLIP (which only processes images)
            tensor = tensor[:, :, 0, :, :]
        
        logger.debug(f"Tensor shape after dimension adjustment: {tensor.shape}")
        
        # Resize to CLIP's expected size and normalize
        image_size = 224
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        tensor = transform(tensor)
        logger.debug(f"Tensor shape after transform: {tensor.shape}")
        
        # Process through CLIP vision model
        with torch.no_grad():
            # Access the vision model
            vision_model = model.vision_model
            
            # Process through vision model
            logger.debug("Running CLIP vision model")
            outputs = vision_model(tensor, output_hidden_states=True)
            
            # Extract features from penultimate layer
            features = outputs.hidden_states[-2]
            logger.debug(f"CLIP features shape: {features.shape}")
        
        # Return features and metadata
        result = {
            "latents": features,
            "position": config.get("position", 0),
            "frames": features.shape[1] if len(features.shape) > 2 else 1
        }
        logger.debug(f"CLIP encoding complete: features shape={features.shape}")
        return result
        
    except Exception as e:
        logger.error(f"CLIP encoding error: {e}", exc_info=True)
        logger.error(f"Tensor shape: {tensor.shape if 'tensor' in locals() else 'unknown'}")
        logger.error(f"CLIP model type: {type(model)}")
        logger.error(f"Config: {config}")
        # Re-raise for proper error handling
        raise
