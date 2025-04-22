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
    
    # Move to model device
    device = model.device
    tensor = tensor.to(device)
    
    # Apply repetition if specified
    repeat = config.get("repeat", 1)
    if repeat > 1 and len(tensor.shape) >= 5:  # B, C, F, H, W
        # Determine which frames to repeat
        frame_dim = 2
        frames = tensor.shape[frame_dim]
        
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
    
    # Encode through VAE
    with torch.no_grad():
        # Encode tensor through VAE
        vae_output = model.encode(tensor)
        
        # Handle DiagonalGaussianDistribution output
        if isinstance(vae_output, DiagonalGaussianDistribution):
            latents = vae_output.sample()
        else:
            latents = vae_output
        
        # Apply VAE scaling
        scale_factor = 1.0 / getattr(model.config, "scaling_factor", 0.18215)
        latents = latents * scale_factor
    
    # Return latents and metadata
    return {
        "latents": latents,
        "position": config.get("position", 0),
        "frames": latents.shape[2] if len(latents.shape) > 3 else 1
    }


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
    
    # Move to model device
    device = model.device
    tensor = tensor.to(device)
    
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
    
    # Process through CLIP vision model
    with torch.no_grad():
        # Access the vision model
        vision_model = model.vision_model
        
        # Process through vision model
        outputs = vision_model(tensor, output_hidden_states=True)
        
        # Extract features from penultimate layer
        features = outputs.hidden_states[-2]
    
    # Return features and metadata
    return {
        "latents": features,
        "position": config.get("position", 0),
        "frames": features.shape[1] if len(features.shape) > 2 else 1
    }
