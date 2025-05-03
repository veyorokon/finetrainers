"""Reference image processors for creating control and CLIP embeddings."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from diffusers.utils import load_image
from PIL import Image

from finetrainers.logging import get_logger
from finetrainers.processors.base import ProcessorMixin

logger = get_logger()


def _crop_and_resize_pad(image, height, width, resize_mode="bicubic"):
    """Center crop and resize image with padding to maintain aspect ratio.
    Uses height, width order to match video_resolution_buckets convention.
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL for processing
        if image.dim() == 3:  # [C, H, W]
            image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 127.5 + 127.5).astype("uint8"))
        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")
    
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Determine aspect ratio
    target_ratio = width / height
    orig_ratio = orig_width / orig_height
    
    if orig_ratio > target_ratio:
        # Image is wider than target ratio
        new_width = int(orig_height * target_ratio)
        new_height = orig_height
        left = (orig_width - new_width) // 2
        image = image.crop((left, 0, left + new_width, new_height))
    else:
        # Image is taller than target ratio
        new_width = orig_width
        new_height = int(orig_width / target_ratio)
        top = (orig_height - new_height) // 2
        image = image.crop((0, top, new_width, top + new_height))
    
    # Resize to target dimensions
    image = image.resize((width, height), getattr(Image, resize_mode.upper()))
    return image


def _pil_to_tensor(image):
    """Convert PIL image to normalized tensor in range [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)


class ReferenceToControlProcessor(ProcessorMixin):
    """Processor that converts reference images to control inputs during preprocessing.
    
    This processor runs before the WanLatentEncodeProcessor in the control processing chain.
    It ensures that control inputs are created from reference images during preprocessing,
    avoiding timing issues in the pipeline.
    """
    
    def __init__(self, output_names: List[str], reference_config: Dict[str, Any] = None, 
                 input_names: Optional[Dict[str, str]] = None):
        super().__init__()
        self.output_names = output_names
        self.reference_config = reference_config
        # Default input names mapping
        self.input_names = input_names or {}
        
    def _preprocess_references(self, references: Dict[str, str]) -> List[Dict[str, Any]]:
        """Convert raw references to pre-processed vae_references format.
        
        Args:
            references: Dictionary mapping reference types to file paths
            
        Returns:
            List of processed reference images with repeat counts
        """
        if not references:
            return []
            
        processed_references = []
        
        # Get config values
        vae_resolution = self.reference_config["vae_resolution"]
        reference_order = self.reference_config["reference_order"]
        repeat_frames = self.reference_config["repeat_frames"]
        
        # Process reference images in specified order
        for idx, ref_type in enumerate(reference_order):
            if ref_type in references:
                # Load the reference image
                ref_path = references[ref_type]
                ref_image = load_image(ref_path)
                
                # Get repetition count (or default to 1)
                repeat = repeat_frames[idx] if idx < len(repeat_frames) else 1
                
                # Process for VAE
                vae_image = _crop_and_resize_pad(
                    ref_image,
                    height=vae_resolution[0],  # [height, width] format
                    width=vae_resolution[1]
                )
                
                processed_references.append({"image": vae_image, "repeat": repeat, "type": ref_type})
                
        return processed_references
    
    def forward(
        self,
        references: Optional[Dict[str, str]] = None,
        vae_references: Optional[List[Dict[str, Any]]] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Process reference images into control inputs.
        
        Args:
            references: Dictionary mapping reference types to file paths
            vae_references: Pre-processed reference images from dataset
            image: Existing image input (passed through if no references)
            video: Existing video input (passed through if no references)
            
        Returns:
            Dictionary with control_image or control_video_list for latent encoding.
            The video_list is always a list of tensors, even if it's just one tensor.
        """
        logger.info(f"ReferenceToControlProcessor input: references={references is not None}, "
                  f"vae_references={vae_references is not None}, "
                  f"image={image is not None}, video={video is not None}")
        
        # Check if we have existing control inputs
        has_control = "control_image" in kwargs or "control_video" in kwargs
        logger.info(f"Has existing control inputs: {has_control}")
        
        # Get vae_combine setting from reference_config
        vae_combine = self.reference_config.get("vae_combine", "before")
        logger.info(f"Using vae_combine method: {vae_combine}")
        
        # Log all available kwargs 
        logger.info(f"Available kwargs: {list(kwargs.keys())}")
        
        # Check for vae_references in all possible locations
        if "vae_references" in kwargs:
            logger.info(f"vae_references found in kwargs with {len(kwargs['vae_references'])} items")
            
        # Clone the existing inputs rather than passing them directly
        # This avoids the warning about overwriting existing values
        image_out = image.clone() if image is not None else None
        video_out = video.clone() if video is not None else None
        
        # If we already have control inputs, don't process references
        if has_control:
            logger.info(f"Using existing control inputs")
            if "control_video" in kwargs:
                # Wrap the existing control video in a list for consistency
                video_list = [kwargs["control_video"]]
                return {self.output_names[0]: image_out, self.output_names[1]: video_list}
            return {self.output_names[0]: image_out, self.output_names[1]: video_out}
        
        # Convert raw references to pre-processed format if needed
        if references and not vae_references:
            logger.info(f"Converting raw references to vae_references format")
            vae_references = self._preprocess_references(references)
            
        # Process references if we have them
        if vae_references and len(vae_references) > 0:
            logger.info(f"Processing {len(vae_references)} reference images")
            
            if vae_combine == "before":
                # For "before" mode: Combine all references into a single video tensor
                logger.info("Using 'before' VAE combine mode: combining references before VAE")
                
                # Create a sequence of frames with specified repetitions
                frames = []
                for ref_data in vae_references:
                    ref_image = ref_data["image"]
                    repeat_count = ref_data["repeat"]
                    logger.info(f"  Reference with repeat count {repeat_count}")
                    
                    # Convert PIL to tensor if needed
                    if not isinstance(ref_image, torch.Tensor):
                        ref_tensor = _pil_to_tensor(ref_image)
                    else:
                        ref_tensor = ref_image
                        
                    frames.extend([ref_tensor] * repeat_count)
                
                if frames:
                    logger.info(f"Creating combined control video with {len(frames)} frames")
                    
                    # Stack frames to create video [T, C, H, W]
                    control_video = torch.stack(frames, dim=0)
                    # Add batch dimension [B, T, C, H, W]
                    control_video = control_video.unsqueeze(0)
                    
                    logger.info(f"Created control video with shape {control_video.shape}")
                    
                    # Return a list containing the single combined video
                    video_list = [control_video]
                    logger.info(f"Returning list with 1 combined video tensor")
                    
                    # Return the control video list using output_names for key mapping
                    result = {self.output_names[0]: None, self.output_names[1]: video_list}
                    logger.info(f"Final output keys: {list(result.keys())}")
                    return result
            else:
                # For "after" mode: Create a list of individual reference tensors
                logger.info("Using 'after' VAE combine mode: keeping references as separate tensors")
                
                # Create a list of individual reference tensors
                video_list = []
                for ref_data in vae_references:
                    ref_image = ref_data["image"]
                    repeat_count = ref_data["repeat"]
                    ref_type = ref_data.get("type", "unknown")
                    logger.info(f"  Reference {ref_type} with repeat count {repeat_count}")
                    
                    # Convert PIL to tensor if needed
                    if not isinstance(ref_image, torch.Tensor):
                        ref_tensor = _pil_to_tensor(ref_image)
                    else:
                        ref_tensor = ref_image
                    
                    # Create a video tensor with repeated frames
                    frames = [ref_tensor] * repeat_count
                    single_video = torch.stack(frames, dim=0).unsqueeze(0)  # [1, T, C, H, W]
                    logger.info(f"  Created video tensor with shape {single_video.shape}")
                    
                    video_list.append(single_video)
                
                if video_list:
                    logger.info(f"Returning list with {len(video_list)} separate video tensors")
                    
                    # Return the control video list using output_names for key mapping
                    result = {self.output_names[0]: None, self.output_names[1]: video_list}
                    logger.info(f"Final output keys: {list(result.keys())}")
                    return result
        
        # If no references were processed, return the original inputs wrapped in a list
        logger.info(f"No references processed, returning original inputs")
        video_list = [video_out] if video_out is not None else None
        return {self.output_names[0]: image_out, self.output_names[1]: video_list}


class WanReferenceLatentEncodeProcessor(ProcessorMixin):
    """
    Specialized processor for reference training that handles lists of video tensors.
    
    This processor is used with the ReferenceToControlProcessor which always returns
    a list of video tensors (either a list with one pre-combined tensor for 'before'
    mode, or a list of individual tensors for 'after' mode).
    
    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are:
            - latents: The latents of the input video list
            - latents_mean: The mean of the latent distribution
            - latents_std: The std of the latent distribution
    """
    
    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 3, "WanReferenceLatentEncodeProcessor requires exactly 3 output names"
        logger.info("Initializing WanReferenceLatentEncodeProcessor")
        
    def forward(
        self,
        vae: Any,  # AutoencoderKLWan
        image: Optional[torch.Tensor] = None,
        video: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a list of video tensors through the VAE.
        
        Args:
            vae: The VAE model
            image: Image tensor (ignored, for compatibility with interface)
            video: List of video tensors to encode (from ReferenceToControlProcessor)
            generator: Optional random generator for posterior sampling
            compute_posterior: Whether to compute the posterior distribution
            
        Returns:
            Dictionary with latents, latents_mean, and latents_std
        """
        device = vae.device
        dtype = vae.dtype
        
        # Handle standard interface parameters but expect video to be a list
        if image is not None:
            logger.warning("Image input provided but ignored - WanReferenceLatentEncodeProcessor uses video list")
        
        if video is None:
            logger.error("No video list provided")
            raise ValueError("WanReferenceLatentEncodeProcessor requires a list of video tensors")
            
        logger.info(f"WanReferenceLatentEncodeProcessor processing list of {len(video)} videos")
        
        # Skip None entries
        video = [v for v in video if v is not None]
        if not video:
            logger.error("No valid videos in list to process")
            raise ValueError("No valid videos to process in video list")
        
        # Process each video tensor and encode it
        latents_list = []
        for i, video_tensor in enumerate(video):
            # Fix dimensionality if needed
            if video_tensor.ndim == 4:  # [F, C, H, W]
                logger.info(f"Converting 4D video tensor {i} with shape {video_tensor.shape} to 5D")
                video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension [1, F, C, H, W]
            
            assert video_tensor.ndim == 5, f"Expected 5D tensor, got {video_tensor.ndim}D tensor for video {i}"
            
            # Process video
            logger.info(f"Processing video {i} with shape {video_tensor.shape}")
            video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
            video_tensor = video_tensor.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
            
            # Encode with VAE
            if compute_posterior:
                logger.info(f"Computing posterior with VAE for video {i}")
                video_latents = vae.encode(video_tensor).latent_dist.sample(generator=generator)
                video_latents = video_latents.to(dtype=dtype)
            else:
                logger.info(f"Encoding video {i} with VAE without posterior sampling")
                video_moments = vae._encode(video_tensor)
                video_latents = video_moments.to(dtype=dtype)
            
            logger.info(f"Generated latents for video {i} with shape {video_latents.shape}")
            latents_list.append(video_latents)
        
        # Concatenate latents along frame dimension (dim 2)
        latents = torch.cat(latents_list, dim=2)
        logger.info(f"Combined latents shape: {latents.shape}")
            
        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)

        logger.info(f"Returning outputs with keys: {self.output_names}")
        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}


class ReferenceClipProcessor(ProcessorMixin):
    """
    Processor to encode reference images using CLIP vision model.
    
    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are:
            - image_embeds: The CLIP visual embeddings of the input reference images.
        input_names (`Dict[str, str]`, optional):
            A mapping of input keys to the names expected by the forward method.
    """

    def __init__(self, output_names: List[str], input_names: Optional[Dict[str, str]] = None):
        super().__init__()
        self.output_names = output_names
        self.input_names = input_names or {}
        assert len(self.output_names) == 1

    def forward(
        self,
        image_processor: Any,
        image_encoder: Any,
        images: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process reference images through CLIP vision model.
        
        Args:
            image_processor: CLIP image processor for preprocessing images
            image_encoder: CLIP vision model for encoding images
            images: List of reference images to process
            
        Returns:
            Dictionary with concatenated image embeddings
        """
        device = image_encoder.device
        dtype = image_encoder.dtype
        
        image_embeds_list = []
        
        for image in images:
            # Process image for CLIP
            processed_image = image_processor(images=image, return_tensors="pt").to(device)
            
            # Get visual embedding (using the penultimate layer similar to A2)
            with torch.no_grad():
                image_embeds = image_encoder(**processed_image, output_hidden_states=True).hidden_states[-2]
                
            # Convert to proper dtype
            image_embeds = image_embeds.to(dtype=dtype)
            image_embeds_list.append(image_embeds)
            
        # Concatenate all reference embeddings along sequence dimension
        all_image_embeds = torch.cat(image_embeds_list, dim=1)
            
        return {self.output_names[0]: all_image_embeds}