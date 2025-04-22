"""Combiner functions for E2V processor types.

This module contains registered combiner functions for combining
features from multiple elements for each processor type.
"""
from typing import Any, Dict, List, Optional, Union

import torch

from finetrainers.logging import get_logger

logger = get_logger()

# Constants for tensor dimensions
BATCH_DIM = 0
CHANNEL_DIM = 1
FRAME_DIM = 2
SEQUENCE_DIM = 1  # For CLIP, sequence is the second dimension

# Registry to store combiner functions
COMBINER_REGISTRY = {}


def register_combiner(name):
    """Register a combiner function for a processor type."""
    def decorator(func):
        COMBINER_REGISTRY[name] = func
        return func
    return decorator


@register_combiner("vae")
def combine_vae_features(features, config=None):
    """Combine VAE features from multiple elements.
    
    For VAE, we:
    1. Sort elements by position
    2. Concatenate along frame dimension (dim=2)
    3. Create and concatenate a frame mask
    4. Return the combined tensor
    
    Args:
        features: Dictionary of features by element name
        config: Configuration dictionary (optional)
        
    Returns:
        Combined tensor
    """
    if not features:
        return None
    
    # Load configuration
    config = config or {}
    concatenate_mask = config.get("concatenate_mask", True)
    
    # Sort elements by position
    sorted_features = sorted(features.values(), key=lambda x: x.get("position", 0))
    
    # Extract latent tensors
    tensors = [f["latents"] for f in sorted_features]
    
    if not tensors:
        return None
    
    # Concatenate along frame dimension
    try:
        combined = torch.cat(tensors, dim=FRAME_DIM)
    except Exception as e:
        logger.error(f"Error concatenating VAE tensors: {e}")
        logger.error(f"Tensor shapes: {[t.shape for t in tensors]}")
        raise
    
    # Create frame mask if needed
    if concatenate_mask:
        # Create zeros tensor with same shape
        mask = torch.zeros_like(combined)
        
        # Set mask to 1 for actual frames (not padding)
        num_frames = combined.shape[FRAME_DIM]
        mask[:, :, :num_frames] = 1.0
        
        # Concatenate mask with latents along channel dimension
        combined = torch.cat([mask, combined], dim=CHANNEL_DIM)
    
    return combined


@register_combiner("clip")
def combine_clip_features(features, config=None):
    """Combine CLIP features from multiple elements.
    
    For CLIP, we:
    1. Sort elements by position
    2. Concatenate along sequence dimension (dim=1)
    3. Return the combined tensor
    
    Args:
        features: Dictionary of features by element name
        config: Configuration dictionary (optional)
        
    Returns:
        Combined tensor
    """
    if not features:
        return None
    
    # Sort elements by position
    sorted_features = sorted(features.values(), key=lambda x: x.get("position", 0))
    
    # Extract latent tensors
    tensors = [f["latents"] for f in sorted_features]
    
    if not tensors:
        return None
    
    # Concatenate along sequence dimension
    try:
        combined = torch.cat(tensors, dim=SEQUENCE_DIM)
    except Exception as e:
        logger.error(f"Error concatenating CLIP tensors: {e}")
        logger.error(f"Tensor shapes: {[t.shape for t in tensors]}")
        raise
    
    return combined


def get_encoder(name):
    """Get encoder function by name."""
    from .encoders import ENCODER_REGISTRY
    
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"No encoder registered for processor: {name}")
    
    return ENCODER_REGISTRY[name]


def get_combiner(name):
    """Get combiner function by name."""
    if name not in COMBINER_REGISTRY:
        raise ValueError(f"No combiner registered for processor: {name}")
    
    return COMBINER_REGISTRY[name]
