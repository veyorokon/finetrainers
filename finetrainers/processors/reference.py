"""Reference image processors for creating control and CLIP embeddings."""

from typing import Any, Dict, List, Optional, Union

import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers.utils import load_image

from finetrainers.logging import get_logger
from finetrainers.processors.base import ProcessorMixin

logger = get_logger()


def _crop_and_resize_pad(image, height, width, resize_mode="bicubic"):
    """Center crop and resize image with padding to maintain aspect ratio."""
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
        self.reference_config = reference_config or {
            "vae_resolution": [854, 480],
            "reference_order": ["object", "background"],
            "repeat_frames": [4, 1]
        }
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
                    height=vae_resolution[1],
                    width=vae_resolution[0]
                )
                
                processed_references.append({"image": vae_image, "repeat": repeat})
                
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
            Dictionary with control_image or control_video for latent encoding
        """
        logger.info(f"ReferenceToControlProcessor input: references={references is not None}, "
                  f"vae_references={vae_references is not None}, "
                  f"image={image is not None}, video={video is not None}")
        
        # Check if we have existing control inputs
        has_control = "control_image" in kwargs or "control_video" in kwargs
        logger.info(f"Has existing control inputs: {has_control}")
        
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
            return {self.output_names[0]: image_out, self.output_names[1]: video_out}
        
        # Convert raw references to pre-processed format if needed
        if references and not vae_references:
            logger.info(f"Converting raw references to vae_references format")
            vae_references = self._preprocess_references(references)
            
        # Process references if we have them
        if vae_references and len(vae_references) > 0:
            logger.info(f"Processing {len(vae_references)} reference images")
            
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
                logger.info(f"Creating control video with {len(frames)} frames")
                # Stack frames to create video [T, C, H, W]
                control_video = torch.stack(frames, dim=0)
                # Add batch dimension [B, T, C, H, W]
                control_video = control_video.unsqueeze(0)
                # The base processor expects [B, F, C, H, W] which it will then permute
                # Don't permute here, let the base WanLatentEncodeProcessor handle it
                
                logger.info(f"Created control video with shape {control_video.shape}")
                
                # Return the control video using defined output_names
                # These should be unique names that don't conflict with existing inputs  
                logger.info(f"Using output keys: {self.output_names[0]}, {self.output_names[1]}")
                
                # Return the control video using output_names for key mapping
                result = {self.output_names[0]: None, self.output_names[1]: control_video}
                logger.info(f"Final output keys: {list(result.keys())}")
                return result
        
        # If no references were processed, return the original inputs
        logger.info(f"No references processed, returning original inputs")
        return {self.output_names[0]: image_out, self.output_names[1]: video_out}


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