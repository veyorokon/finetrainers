import numpy as np
import os
import pathlib
from typing import List, Optional, Tuple
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PIL import Image
import torch


def _crop_and_resize_pad(image, height=480, width=720):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        pad = int((((height / width) * image_width) - image_height) / 2.)
        padded_image = np.ones((image_height + pad * 2, image_width, 3), dtype=np.uint8) * 255
        # padded_image = np.zeros((image_height + pad * 2, image_width, 3), dtype=np.uint8)
        padded_image[pad:pad+image_height, :] = image
        image = Image.fromarray(padded_image).resize((width, height))
    else:
        pad = int((((width / height) * image_height) - image_width) / 2.)
        padded_image = np.ones((image_height, image_width + pad * 2, 3), dtype=np.uint8) * 255
        # padded_image = np.zeros((image_height, image_width + pad * 2, 3), dtype=np.uint8) 
        padded_image[:, pad:pad+image_width] = image
        image = Image.fromarray(padded_image).resize((width, height))
    return image 


def _crop_and_resize(image, height=512, width=512):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width/width*height)
        top = (image_height - croped_height) // 2
        image = image[top:top+croped_height, :]
        image = Image.fromarray(image).resize((width, height))

    return image


def _scale_height_and_pad(image, height=480, width=720, scale_factor=0.5, pad_value=255):
    """
    Scales the image to a percentage of its height and resizes to specified dimensions.
    The remaining space is filled with padding.
    
    Args:
        image: PIL Image or numpy array
        height: Target height of the output image
        width: Target width of the output image
        scale_factor: Float between 0 and 1, percentage of height to scale to
        pad_value: Value to use for padding (255 for white, 0 for black)
        
    Returns:
        PIL Image with scaled content and fixed dimensions
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        
    # First resize to target dimensions while preserving aspect ratio
    image_height, image_width, channels = image.shape
    if image_height / image_width < height / width:
        pad = int((((height / width) * image_width) - image_height) / 2.)
        padded_image = np.ones((image_height + pad * 2, image_width, channels), dtype=np.uint8) * pad_value
        padded_image[pad:pad+image_height, :] = image
        image = Image.fromarray(padded_image).resize((width, height))
    else:
        pad = int((((width / height) * image_height) - image_width) / 2.)
        padded_image = np.ones((image_height, image_width + pad * 2, channels), dtype=np.uint8) * pad_value
        padded_image[:, pad:pad+image_width] = image
        image = Image.fromarray(padded_image).resize((width, height))
    
    # Convert back to numpy for scaling
    image = np.array(image)
    
    # Calculate how much smaller the scaled content should be
    new_height = int(height * scale_factor)
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    
    # Create a blank canvas of the target size
    result_image = np.ones((height, width, 3), dtype=np.uint8) * pad_value
    
    # Resize the image to the scaled dimensions
    scaled_content = Image.fromarray(image).resize((new_width, new_height), Image.LANCZOS)
    scaled_content = np.array(scaled_content)
    
    # Calculate position to place scaled content (centered)
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2
    
    # Place the scaled content in the center
    result_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled_content
    
    return Image.fromarray(result_image)
    

def write_mp4(video_path, samples, fps=14, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate, 
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])


def create_channel_frame_grid(
    latents: torch.Tensor,
    output_dir: str,
    filename: str = "latent_grid.png",
    spacing: int = 2,
    group_sizes: Optional[List[int]] = None,
) -> str:
    """
    Creates a visualization grid of latent space with channels as columns and frames as rows.
    
    Args:
        latents: Tensor of shape [B, C, T, H, W] (batch, channels, frames, height, width)
        output_dir: Directory to save the output image
        filename: Name of the output file
        spacing: Spacing between cells in the grid
        group_sizes: Optional list of channel group sizes to visually separate in the output
                    (e.g. [4, 16, 16] for mask(4), content(16), control(16))
    
    Returns:
        Path to the saved image file
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
    
    print(f"Creating grid with {num_channels} channels x {num_frames} frames")
    
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
            # Get frame data and convert to float32 before numpy conversion
            data = latent_data[c, t].to(torch.float32).numpy()
            
            # Direct raw value visualization with grayscale for clearer interpretation
            # Shift from [-1,1] to [0,1] range for direct visualization
            norm_data = (data + 1.0) * 0.5
            
            # Make sure we clamp the range to [0,1]
            norm_data = np.clip(norm_data, 0, 1)
            
            # Convert to grayscale (0 = black, 1 = white)
            # Create RGB with uniform values (grayscale)
            grayscale = (norm_data * 255).astype(np.uint8)
            colored_data = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
            colored_data[..., 0] = grayscale  # R
            colored_data[..., 1] = grayscale  # G
            colored_data[..., 2] = grayscale  # B
            colored_data[..., 3] = 255        # A (fully opaque)
            
            # Add debugging info
            if t == 0 and c == 0:  # Log only for first frame of first channel
                print(f"Raw value range: min={data.min():.4f}, max={data.max():.4f}")
                print(f"Normalized range: min={norm_data.min():.4f}, max={norm_data.max():.4f}")
            
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
    
    print(f"Saved latent grid to {file_path}")
    
    return str(file_path)
