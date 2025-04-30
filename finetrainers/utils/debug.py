"""Minimal debugging utilities for latent visualization."""

import os
import pathlib
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib import cm

# Ensure NumPy is imported as np

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


def create_channel_frame_grid(
    latents: torch.Tensor,
    output_dir: str,
    filename: str = "latent_grid.png",
    spacing: int = 2,
    group_sizes: Optional[List[int]] = None,
) -> str:
    """Simple visualization of latents as a grid of frames x channels."""
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Move to CPU
    latents = latents.detach().cpu()
    
    # Only support video latents (5D) with this visualization
    if latents.dim() != 5:
        raise ValueError("Channel-frame grid only supports 5D latents [B, C, T, H, W]")
    
    # Extract dimensions - taking only first batch
    batch_size, num_channels, num_frames, height, width = latents.shape
    latent_data = latents[0]  # [C, T, H, W]
    
    logger.info(f"Grid with {num_channels} channels x {num_frames} frames")
    
    # Get colormap
    cmap = cm.get_cmap('viridis')
    
    # Set spacing
    spacing = 2
    
    # Calculate total size
    grid_width = num_channels * width + (num_channels - 1) * spacing
    grid_height = num_frames * height + (num_frames - 1) * spacing
    
    # Create empty grid
    grid_img = Image.new('RGB', (grid_width, grid_height), color='black')
    
    # Generate one column per channel, with frames as rows
    for c in range(num_channels):
        # Create a column for this channel
        col_width = width
        col_height = num_frames * height + (num_frames - 1) * spacing
        col_img = Image.new('RGB', (col_width, col_height), color='black')
        
        # Process each frame for this channel
        for t in range(num_frames):
            # Get frame data
            data = latent_data[c, t].numpy()
            
            # Normalize to [0,1] with better handling of padding
            # Check if this is mostly padding (near-zero values)
            is_padding = np.abs(data).max() < 1e-4
            
            if is_padding:
                # If it's padding, make it pure black
                norm_data = np.zeros_like(data)
                # Use a different colormap for padding
                colored_data = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
            else:
                # For active frames, use robust normalization
                min_val, max_val = data.min(), data.max()
                abs_max = max(abs(min_val), abs(max_val))
                
                # If very small values, treat as zero
                if abs_max < 1e-3:
                    norm_data = np.zeros_like(data)
                else:
                    # Center at zero and scale by absolute maximum
                    norm_data = np.clip(data / abs_max, -1, 1) * 0.5 + 0.5
                
                # Apply colormap
                colored_data = (cmap(norm_data) * 255).astype(np.uint8)
            
            # Create image
            frame_img = Image.fromarray(colored_data)
            
            # Paste into column
            y_pos = t * (height + spacing)
            col_img.paste(frame_img, (0, y_pos))
        
        # Paste column into grid
        x_pos = c * (width + spacing)
        grid_img.paste(col_img, (x_pos, 0))
    
    # Save the grid
    file_path = output_path / filename
    grid_img.save(file_path)
    
    logger.info(f"Saved grid to {file_path}")
    
    return str(file_path)