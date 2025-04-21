from typing import List, Literal, Tuple

import torch
import torch.nn.functional as F


def center_crop_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Crop the center of the image to the target size.
    
    Args:
        image: Input tensor [C, H, W]
        size: Target (height, width)
    
    Returns:
        Center-cropped tensor [C, target_h, target_w]
    """
    num_channels, height, width = image.shape
    crop_h, crop_w = size
    if height < crop_h or width < crop_w:
        raise ValueError(f"Image size {(height, width)} is smaller than the target size {size}.")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[:, top : top + crop_h, left : left + crop_w]


def resize_crop_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize the image to cover the target size, then crop the center.
    
    Args:
        image: Input tensor [C, H, W]
        size: Target (height, width)
    
    Returns:
        Resized and cropped tensor [C, target_h, target_w]
    """
    num_channels, height, width = image.shape
    target_h, target_w = size
    scale = max(target_h / height, target_w / width)
    new_h, new_w = int(height * scale), int(width * scale)
    image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    return center_crop_image(image, size)


def bicubic_resize_image(image: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize the image using bicubic interpolation.
    
    Args:
        image: Input tensor [C, H, W]
        size: Target (height, width)
        
    Returns:
        Resized tensor [C, target_h, target_w]
    """
    return F.interpolate(image.unsqueeze(0), size=size, mode="bicubic", align_corners=False)[0]


def find_nearest_resolution_image(image: torch.Tensor, resolution_buckets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Find the resolution bucket that best matches the image's aspect ratio.
    
    Args:
        image: Input tensor [C, H, W]
        resolution_buckets: List of (height, width) tuples
        
    Returns:
        The (height, width) tuple from resolution_buckets that best matches the image's aspect ratio
    """
    num_channels, height, width = image.shape
    aspect_ratio = width / height

    def aspect_ratio_diff(bucket):
        return abs((bucket[1] / bucket[0]) - aspect_ratio), (-bucket[0], -bucket[1])

    return min(resolution_buckets, key=aspect_ratio_diff)


def resize_to_nearest_bucket_image(
    image: torch.Tensor,
    resolution_buckets: List[Tuple[int, int]],
    resize_mode: Literal["center_crop", "resize_crop", "bicubic"] = "bicubic",
) -> torch.Tensor:
    """Resize the image to the nearest resolution bucket.
    
    Args:
        image: Input tensor [C, H, W]
        resolution_buckets: List of (height, width) tuples
        resize_mode: The resize mode to use
        
    Returns:
        Resized tensor [C, target_h, target_w]
    """
    target_size = find_nearest_resolution_image(image, resolution_buckets)

    if resize_mode == "center_crop":
        return center_crop_image(image, target_size)
    elif resize_mode == "resize_crop":
        return resize_crop_image(image, target_size)
    elif resize_mode == "bicubic":
        return bicubic_resize_image(image, target_size)
    else:
        raise ValueError(
            f"Invalid resize_mode: {resize_mode}. Choose from 'center_crop', 'resize_crop', or 'bicubic'."
        )


def trim_transparency(
    image: torch.Tensor, 
    alpha_threshold: float = 0.1,
    padding_buffer: int = 0
) -> torch.Tensor:
    """Trim transparent pixels from an image.
    
    Args:
        image: Input tensor with alpha channel [C, H, W] or [B, C, H, W]
        alpha_threshold: Threshold for considering pixels non-transparent
        padding_buffer: Number of pixels to pad around trimmed image
        
    Returns:
        Trimmed tensor with same number of dimensions
    """
    # Handle batch dimension
    has_batch_dim = image.dim() == 4
    if has_batch_dim:
        batch_size, num_channels, height, width = image.shape
        if num_channels < 4:  # No alpha channel
            return image
            
        # Process each image in batch
        trimmed_images = []
        for i in range(batch_size):
            img = image[i]
            alpha = img[3]  # Get alpha channel
            # Find non-transparent pixels
            non_transparent = alpha > alpha_threshold
            if non_transparent.any():
                # Get bounding box of non-transparent pixels
                rows = torch.any(non_transparent, dim=1)
                cols = torch.any(non_transparent, dim=0)
                
                # Find min/max indices
                rmin, rmax = torch.where(rows)[0][[0, -1]]
                cmin, cmax = torch.where(cols)[0][[0, -1]]
                
                # Add padding buffer
                if padding_buffer > 0:
                    rmin = max(0, rmin - padding_buffer)
                    rmax = min(height - 1, rmax + padding_buffer)
                    cmin = max(0, cmin - padding_buffer)
                    cmax = min(width - 1, cmax + padding_buffer)
                
                # Crop the image
                trimmed = img[:, rmin:rmax+1, cmin:cmax+1]
            else:
                # If all transparent, keep original
                trimmed = img
            trimmed_images.append(trimmed)
        
        # Pad to same size if needed for batching
        max_h = max(img.shape[1] for img in trimmed_images)
        max_w = max(img.shape[2] for img in trimmed_images)
        
        # Pad all to same size
        padded_images = []
        for img in trimmed_images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            if pad_h > 0 or pad_w > 0:
                padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
                padded_images.append(padded)
            else:
                padded_images.append(img)
        
        # Stack back to batch
        return torch.stack(padded_images)
    else:
        # Single image processing
        num_channels, height, width = image.shape
        if num_channels < 4:  # No alpha channel
            return image
            
        alpha = image[3]  # Get alpha channel
        # Find non-transparent pixels
        non_transparent = alpha > alpha_threshold
        if non_transparent.any():
            # Get bounding box of non-transparent pixels
            rows = torch.any(non_transparent, dim=1)
            cols = torch.any(non_transparent, dim=0)
            
            # Find min/max indices
            rmin, rmax = torch.where(rows)[0][[0, -1]]
            cmin, cmax = torch.where(cols)[0][[0, -1]]
            
            # Add padding buffer
            if padding_buffer > 0:
                rmin = max(0, rmin - padding_buffer)
                rmax = min(height - 1, rmax + padding_buffer)
                cmin = max(0, cmin - padding_buffer)
                cmax = min(width - 1, cmax + padding_buffer)
            
            # Crop the image
            return image[:, rmin:rmax+1, cmin:cmax+1]
        else:
            # If all transparent, keep original
            return image


def letterbox_image(
    image: torch.Tensor, 
    size: Tuple[int, int], 
    padding_color: float = 0.0,
    trim_alpha: bool = False,
    alpha_threshold: float = 0.1,
    padding_buffer: int = 0
) -> torch.Tensor:
    """Letterbox an image to fit target size while maintaining aspect ratio.
    
    Args:
        image: Input tensor [C, H, W] or [B, C, H, W]
        size: Target (height, width)
        padding_color: Value to use for padding (default: 0.0 for black)
        trim_alpha: Whether to trim transparent pixels before resizing
        alpha_threshold: Threshold for considering pixels non-transparent
        padding_buffer: Number of pixels to pad around trimmed image
        
    Returns:
        Letterboxed tensor of shape [C, target_h, target_w] or [B, C, target_h, target_w]
    """
    # First trim transparency if requested
    if trim_alpha:
        has_alpha = (image.shape[1] == 4) if image.dim() == 4 else (image.shape[0] == 4)
        if has_alpha:
            image = trim_transparency(image, alpha_threshold, padding_buffer)
    
    # Get dimensions
    has_batch_dim = image.dim() == 4
    if has_batch_dim:
        batch_size, num_channels, height, width = image.shape
    else:
        num_channels, height, width = image.shape
    
    target_h, target_w = size
    
    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_h / height, target_w / width)
    
    # Calculate new size after scaling
    new_h, new_w = int(height * scale), int(width * scale)
    
    # Resize image to new size
    mode = "bicubic" if num_channels == 3 else "nearest"
    if has_batch_dim:
        resized = F.interpolate(image, size=(new_h, new_w), mode=mode, align_corners=False if mode == "bicubic" else None)
    else:
        resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode=mode, align_corners=False if mode == "bicubic" else None).squeeze(0)
    
    # Calculate padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(resized, padding, mode="constant", value=padding_color)
