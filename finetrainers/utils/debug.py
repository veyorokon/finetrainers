"""Minimal debugging utilities for latent visualization."""

import os
import pathlib
from typing import List, Optional, Tuple

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


def create_channel_frame_grid(
    latents: torch.Tensor,
    output_dir: str,
    filename: str = "latent_grid.png",
    spacing: int = 2,
    group_sizes: Optional[List[int]] = None,
) -> str:
    """
    Create a grid visualization with channels as columns and frames as rows.
    
    Args:
        latents: Tensor of shape [B, C, T, H, W] with channels and frames
        output_dir: Directory to save visualization
        filename: Output filename 
        spacing: Pixels of spacing between channels/frames
        group_sizes: Optional list of channel group sizes for visual separation
        
    Returns:
        Path to saved grid image
    """
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
    
    # Set default group sizes if not provided
    if group_sizes is None:
        # Default groups: content (16), mask (1), conditioning (16), padding (3)
        group_sizes = [16, 1, 16, 3]
    
    # Normalization done inline for simplicity
    
    # Get colormap
    cmap = cm.get_cmap('viridis')
    
    # Calculate grid dimensions with spacing
    grid_width = num_channels * width + (num_channels - 1) * spacing
    grid_height = num_frames * height + (num_frames - 1) * spacing
    
    # Add extra spacing for channel groups
    if group_sizes:
        # Add wide separators between groups
        group_separators = len(group_sizes) - 1
        grid_width += group_separators * (spacing * 3)
    
    # Create the grid image
    grid_img = Image.new('RGB', (grid_width, grid_height), color='black')
    
    # Track the current x position
    x_pos = 0
    channel_idx = 0
    
    # Process each channel group
    group_start_idx = 0
    for group_idx, group_size in enumerate(group_sizes):
        group_end_idx = group_start_idx + group_size
        
        # Process channels in this group
        for c in range(group_start_idx, group_end_idx):
            if c >= num_channels:
                break
                
            # Process each frame for this channel
            for t in range(num_frames):
                # Get channel data for this frame
                data = latent_data[c, t].numpy()
                
                # Normalize data to [0,1]
                min_val, max_val = data.min(), data.max()
                if min_val == max_val:
                    norm_data = np.zeros_like(data) if min_val < 0 else np.ones_like(data)
                else:
                    norm_data = (data - min_val) / (max_val - min_val)
                
                # Apply colormap
                colored_data = (cmap(norm_data) * 255).astype(np.uint8)
                
                # Create image
                channel_img = Image.fromarray(colored_data)
                
                # Calculate position
                y_pos = t * (height + spacing)
                
                # Paste into grid
                grid_img.paste(channel_img, (x_pos, y_pos))
            
            # Move to next channel
            x_pos += width + spacing
        
        # Move to next group (with extra spacing between groups)
        group_start_idx = group_end_idx
        if group_idx < len(group_sizes) - 1:
            x_pos += spacing * 2  # Extra spacing between groups
    
    # Save the grid
    file_path = output_path / filename
    grid_img.save(file_path)
    
    logger.info(f"Saved channel-frame grid to {file_path}")
    
    return str(file_path)