"""Minimal debugging utilities for latent visualization."""

import os
import pathlib
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from matplotlib import cm

from finetrainers.logging import get_logger

logger = get_logger()


def save_latent_channels(
    latents: torch.Tensor, 
    output_dir: str, 
    prefix: str = "latent",
    channel_indices: Optional[List[int]] = None,
    frame_idx: int = 0,
) -> List[str]:
    """Save selected latent channels as images for visualization."""
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Move to CPU and convert to numpy
    latents = latents.detach().cpu()
    
    # Normalize function
    def normalize_data(data):
        min_val, max_val = data.min(), data.max()
        if min_val == max_val:
            return np.zeros_like(data) if min_val < 0 else np.ones_like(data)
        return (data - min_val) / (max_val - min_val)
    
    # Handle video (5D) and image (4D) latents
    if latents.dim() == 5:  # [B, C, T, H, W]
        # Extract first batch and specified frame
        lat = latents[0, :, frame_idx].numpy()
    else:  # [B, C, H, W]
        # Extract first batch
        lat = latents[0].numpy()
    
    # Get colormap
    cmap = cm.get_cmap('viridis')
    
    # Select channels to visualize
    if channel_indices is None:
        channel_indices = list(range(lat.shape[0]))
    
    saved_paths = []
    
    # Save each selected channel
    for channel_idx in channel_indices:
        # Extract and normalize channel data
        channel_data = lat[channel_idx]
        norm_data = normalize_data(channel_data)
        
        # Apply colormap and convert to image
        colored_data = (cmap(norm_data) * 255).astype(np.uint8)
        img = Image.fromarray(colored_data)
        
        # Save the image
        filename = f"{prefix}_ch{channel_idx}.png"
        output_file = output_path / filename
        img.save(output_file)
        saved_paths.append(str(output_file))
        
        logger.info(f"Saved channel {channel_idx} to {output_file}")
    
    return saved_paths