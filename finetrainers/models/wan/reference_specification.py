import functools
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights
from diffusers import (AutoencoderKLWan, FlowMatchEulerDiscreteScheduler,
                       WanPipeline, WanTransformer3DModel)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import load_image
from transformers import (AutoModel, AutoTokenizer, CLIPImageProcessor,
                          CLIPVisionModel, UMT5EncoderModel)

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.processors import (ProcessorMixin, ReferenceClipProcessor,
                                     ReferenceToControlProcessor, T5Processor)
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import (get_non_null_items,
                                safetensors_torch_save_function)

from .base_specification import WanLatentEncodeProcessor, WanModelSpecification
from .control_specification import WanControlModelSpecification


def apply_reference_frame_conditioning(
    latents: torch.Tensor,
    expected_num_frames: int,
    frame_conditioning_type: str,
    frame_conditioning_index: Optional[int] = None,
    channel_dim: int = 1,
    frame_dim: int = 2,
    concatenate_mask: bool = True,
) -> torch.Tensor:
    """
    Apply frame conditioning for reference model with optional A2-style single-channel mask.
    
    This function first applies standard frame conditioning, then optionally creates a 
    single-channel mask to concatenate, mimicking the A2 paper's approach.
    
    Args:
        latents: Control latents to condition
        expected_num_frames: Number of frames to match
        frame_conditioning_type: Type of conditioning ("index", "full", etc.)
        frame_conditioning_index: Index for index-based conditioning
        channel_dim: Dimension for channels
        frame_dim: Dimension for frames
        concatenate_mask: Whether to concatenate a single-channel mask
        
    Returns:
        Conditioned latents, optionally with single-channel mask concatenated
    """
    from finetrainers.trainer.control_trainer.data import \
        apply_frame_conditioning_on_latents

    # First apply normal frame conditioning (without concatenation)
    masked_latents = apply_frame_conditioning_on_latents(
        latents,
        expected_num_frames,
        channel_dim=channel_dim,
        frame_dim=frame_dim,
        frame_conditioning_type=frame_conditioning_type,
        frame_conditioning_index=frame_conditioning_index,
        concatenate_mask=False,  # Never use built-in concatenate, we handle it ourselves
    )
    
    # If we don't need to concatenate mask, return the masked latents directly
    if not concatenate_mask:
        return masked_latents
    
    # Create a single-channel mask
    mask_shape = list(masked_latents.shape)
    mask_shape[channel_dim] = 1  # Single channel for mask
    mask = torch.zeros(mask_shape, device=masked_latents.device, dtype=masked_latents.dtype)
    
    # Set 1s for reference frames based on frame_conditioning_type
    if frame_conditioning_type == "index":
        frame_index = min(frame_conditioning_index or 0, mask_shape[frame_dim] - 1)
        indexing = [slice(None)] * len(mask_shape)
        indexing[frame_dim] = frame_index
        mask[tuple(indexing)] = 1
    elif frame_conditioning_type == "first_and_last":
        indexing = [slice(None)] * len(mask_shape)
        indexing[frame_dim] = 0
        mask[tuple(indexing)] = 1
        indexing[frame_dim] = -1
        mask[tuple(indexing)] = 1
    elif frame_conditioning_type == "full":
        # Instead of filling all frames with 1s, fill exactly the number 
        # of frames we have in the masked_latents tensor
        num_reference_frames = masked_latents.shape[frame_dim]
        for i in range(num_reference_frames):
            indexing = [slice(None)] * len(mask_shape)
            indexing[frame_dim] = i
            mask[tuple(indexing)] = 1
        # Log mask statistics for debugging
        mask_min = mask.min().item()
        mask_max = mask.max().item()
        mask_mean = mask.mean().item()
        mask_nonzero = (mask > 0).float().sum().item()
        logger.info(f"Mask stats: min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}, " +
                  f"non-zero={mask_nonzero} out of {mask.numel()}")
        logger.info(f"Marked {num_reference_frames} frames as reference frames in mask")
        
    # Concatenate the mask with masked latents (mask FIRST, then latents)
    # This matches the A2 model's inference code ordering
    result = torch.cat([mask, masked_latents], dim=channel_dim)
    logger.info(f"Applied A2-style reference conditioning with single-channel mask: {result.shape}")
    
    # Calculate dynamic padding:
    # - Each VAE latent has 16 channels
    # - We have 1 channel for the mask and 16 for the reference
    # - The transformer expects 36 channels total
    # - So we need 36 - (16 + 1 + 16) = 3 additional padding channels
    
    # Get the current channel count
    vae_channels = 16  # Standard VAE latent channels 
    total_expected = 36  # Transformer input channels
    current_control_channels = result.shape[channel_dim]
    
    # Calculate how many channels we'll have after concatenating with noisy_latents
    total_after_concat = vae_channels + current_control_channels
    
    # Calculate how many padding channels we need
    if total_after_concat < total_expected:
        padding_channels = total_expected - total_after_concat
        padding_shape = list(result.shape)
        padding_shape[channel_dim] = padding_channels
        
        logger.info(f"Dynamically adding {padding_channels} padding channels (current: {current_control_channels}, " +
                  f"total after concat: {total_after_concat}, target: {total_expected})")
        channel_padding = torch.zeros(padding_shape, device=result.device, dtype=result.dtype, requires_grad=True)
        result = torch.cat([result, channel_padding], dim=channel_dim)
    elif total_after_concat > total_expected:
        logger.warning(f"Control latents will have too many channels after concat: {total_after_concat} > {total_expected}")
    
    return result

logger = get_logger()


class WanReferenceModelSpecification(WanControlModelSpecification):
    """
    Model specification for the Wan model with reference-based conditioning (A2 style).
    Extends WanControlModelSpecification to add CLIP visual embedding processing.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        image_encoder_id: Optional[str] = None,
        image_processor_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        image_encoder_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        embedding_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        control_model_processors: List[ProcessorMixin] = None,
        reference_config: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        # Initialize reference config
        self.reference_config = reference_config 
        
        # Create reference-to-control processor if not provided
        if control_model_processors is None:
            # First, create the standard control processor
            standard_control_processor = WanLatentEncodeProcessor(["control_latents", "__drop__", "__drop__"])
            
            # Create a reference processor that runs before it
            reference_processor = ReferenceToControlProcessor(
                ["image", "video"], 
                reference_config=self.reference_config
            )
            
            # Add logging to debug more easily
            logger.info("Initializing control processors:")
            logger.info(f"  Reference processor output names: {reference_processor.output_names}")
            logger.info(f"  Control processor output names: {standard_control_processor.output_names}")
            
            # Use both processors in sequence
            control_model_processors = [reference_processor, standard_control_processor]
        
        # Initialize parent with our processors
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
            condition_model_processors=condition_model_processors,
            latent_model_processors=latent_model_processors,
            control_model_processors=control_model_processors,
        )
        
        # Store image encoder configs
        self.image_encoder_id = image_encoder_id
        self.image_processor_id = image_processor_id
        self.image_encoder_dtype = image_encoder_dtype
        
        # Setup embedding model processors for CLIP encoding
        if embedding_model_processors is None:
            embedding_model_processors = [ReferenceClipProcessor(["encoder_image_embeds"])]
            
        self.embedding_model_processors = embedding_model_processors

    def load_embedding_models(self) -> Dict[str, torch.nn.Module]:
        """Load CLIP vision model and processor for reference-based conditioning"""
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}
        
        if self.image_processor_id is not None:
            image_processor = CLIPImageProcessor.from_pretrained(self.image_processor_id, **common_kwargs)
        else:
            image_processor = CLIPImageProcessor.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_processor", **common_kwargs
            )
            
        if self.image_encoder_id is not None:
            image_encoder = CLIPVisionModel.from_pretrained(
                self.image_encoder_id, torch_dtype=self.image_encoder_dtype, **common_kwargs
            )
        else:
            image_encoder = CLIPVisionModel.from_pretrained(
                self.pretrained_model_name_or_path, 
                subfolder="image_encoder", 
                torch_dtype=self.image_encoder_dtype,
                **common_kwargs
            )
            
        return {"image_processor": image_processor, "image_encoder": image_encoder}
        
    @torch.no_grad()
    def prepare_embeddings(
        self,
        image_processor: CLIPImageProcessor,
        image_encoder: CLIPVisionModel,
        reference_images: List[torch.Tensor],
        **kwargs,
    ) -> Dict[str, Any]:
        """Process reference images through CLIP vision model"""
        conditions = {
            "image_processor": image_processor,
            "image_encoder": image_encoder,
            "images": reference_images,
            **kwargs,
        }
        
        input_keys = set(conditions.keys())
        
        # Process through processors
        for processor in self.embedding_model_processors:
            outputs = processor(**conditions)
            conditions.update(outputs)
            
        # Filter out input keys
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        
        return conditions
        
    def forward(
        self,
        transformer: WanTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        embedding_model_conditions: Dict[str, torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with reference-based conditioning.
        Add reference embeddings to the encoder_hidden_states.
        """
        # Handle case where embedding_model_conditions is not provided
        if embedding_model_conditions is None:
            embedding_model_conditions = {}
            
        # Only modify the condition if we have image embeddings
        if "encoder_image_embeds" in embedding_model_conditions:
            # Get image embeddings
            image_embeds = embedding_model_conditions.pop("encoder_image_embeds")
            
            # Get text embeddings
            text_embeds = condition_model_conditions.pop("encoder_hidden_states")
            
            # Concatenate image and text embeddings
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            
            # Put back in condition_model_conditions
            condition_model_conditions["encoder_hidden_states"] = combined_embeds
        
        # Copy the relevant code from the parent class to handle latents
        from finetrainers.trainer.control_trainer.data import \
            apply_frame_conditioning_on_latents

        compute_posterior = False  # See explanation in prepare_latents
        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            control_latents = latent_model_conditions.pop("control_latents")
        else:
            latents = latent_model_conditions.pop("latents")
            control_latents = latent_model_conditions.pop("control_latents")
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            mu, logvar = torch.chunk(control_latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            control_latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.mode()
            del posterior

            control_posterior = DiagonalGaussianDistribution(control_latents)
            control_latents = control_posterior.mode()
            del control_posterior

        noise = torch.zeros_like(latents).normal_(generator=generator)
        timesteps = (sigmas.flatten() * 1000.0).long()

        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        
        # Let's inspect control_latents before processing
        logger.info(f"Before apply_reference_frame_conditioning, control_latents shape: {control_latents.shape}")
        for f_idx in range(min(8, control_latents.shape[2])):
            frame_data = control_latents[0, :, f_idx]
            non_zero = (frame_data.abs() > 1e-6).float().sum().item()
            logger.info(f"Frame {f_idx}: {non_zero} non-zero values out of {frame_data.numel()}")
            
        # Use our custom reference frame conditioning function
        control_latents = apply_reference_frame_conditioning(
            control_latents,
            noisy_latents.shape[2],
            frame_conditioning_type=self.frame_conditioning_type,
            frame_conditioning_index=self.frame_conditioning_index,
            channel_dim=1,
            frame_dim=2,
            concatenate_mask=self.frame_conditioning_concatenate_mask,
        )
        
        # Concatenate latents along channel dimension
        # Control latents should already have the right number of channels from apply_reference_frame_conditioning
        noisy_latents = torch.cat([noisy_latents, control_latents], dim=1)
        logger.info(f"Final concatenated latents shape for transformer: {noisy_latents.shape}")
        
        # Diagnostic logging - check the mask channel in final concatenated tensor
        # Mask should be at channel 16 (after the content channels)
        mask_channel = noisy_latents[:, 16]
        mask_min = mask_channel.min().item()
        mask_max = mask_channel.max().item()
        mask_mean = mask_channel.mean().item()
        mask_nonzero = (mask_channel > 0.5).float().sum().item()
        logger.info(f"Final mask channel stats: min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}, " +
                  f"non-zero={mask_nonzero} out of {mask_channel.numel()}")
        import os

        # Debug visualization of latent channels - only in the first few steps
        if os.environ.get("REFERENCE_DEBUG_LATENTS") == "1":
            from finetrainers.utils import (create_channel_frame_grid,
                                            save_latent_channels)

            # Create a unique identifier for this step
            step_id = os.environ.get("REFERENCE_DEBUG_STEP_ID", "0")
            
            # Save to debug directory
            output_dir = os.path.join("debug_latents", f"step_{step_id}")
            
            # Save individual channels for quick reference
            # Mask channel (should be at index 16 in final tensor)
            save_latent_channels(noisy_latents, output_dir, "mask", [16])
            
            # Create channel×frame grid visualization
            # Group sizes now match A2 inference order: content (16), mask (1), conditioning (16), padding (3)
            create_channel_frame_grid(
                noisy_latents,
                output_dir,
                filename=f"latent_grid_{step_id}.png",
                group_sizes=[16, 1, 16, 3]
            )
            
            # Increment step counter
            next_step = int(step_id) + 1
            os.environ["REFERENCE_DEBUG_STEP_ID"] = str(next_step)
        
        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)

        pred = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            return_dict=False,
        )[0]
        target = FF.flow_match_target(noise, latents)

        return pred, target, sigmas
    
    def validation(
        self,
        pipeline: WanPipeline,
        prompt: str,
        reference_images: List[torch.Tensor] = None,
        vae_references: List[Dict[str, Any]] = None,
        references: Dict[str, str] = None,
        control_image: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        frame_conditioning_type: str = "full",
        frame_conditioning_index: int = 0,
        **kwargs,
    ) -> List[ArtifactType]:
        """
        Extend validation to include reference images for A2-style conditioning
        """
        from finetrainers.processors.reference import \
            ReferenceToControlProcessor
        from finetrainers.trainer.control_trainer.data import \
            apply_frame_conditioning_on_latents

        with torch.no_grad():
            dtype = pipeline.vae.dtype
            device = pipeline._execution_device
            in_channels = self.transformer_config.in_channels  # We need to use the original in_channels
            latents = pipeline.prepare_latents(1, in_channels, height, width, num_frames, dtype, device, generator)
            latents_mean = (
                torch.tensor(self.vae_config.latents_mean)
                .view(1, self.vae_config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_config.latents_std).view(1, self.vae_config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )

            # Create control video from references if needed
            if (control_image is None and control_video is None) and (vae_references or references):
                # Create a reference processor for validation
                reference_processor = ReferenceToControlProcessor(
                    ["image", "video"], 
                    reference_config=self.reference_config
                )
                
                # Process references - this handles vae_references and raw references
                logger.info(f"Using ReferenceToControlProcessor during validation")
                result = reference_processor(
                    references=references,
                    vae_references=vae_references
                )
                
                # Get the control video
                control_video = result.get("video")
                if control_video is not None:
                    logger.info(f"Created control video with shape {control_video.shape}")
            
            # Process existing control image/video
            if control_image is not None:
                control_video = pipeline.video_processor.preprocess(
                    control_image, height=height, width=width
                ).unsqueeze(2)
            elif control_video is not None:
                control_video = pipeline.video_processor.preprocess_video(control_video, height=height, width=width)

            # Convert to latents
            control_video = control_video.to(device=device, dtype=dtype)
            control_latents = pipeline.vae.encode(control_video).latent_dist.mode()
            control_latents = self._normalize_latents(control_latents, latents_mean, latents_std)
            control_latents = apply_reference_frame_conditioning(
                control_latents,
                latents.shape[2],
                frame_conditioning_type=frame_conditioning_type,
                frame_conditioning_index=frame_conditioning_index,
                channel_dim=1,
                frame_dim=2,
                concatenate_mask=self.frame_conditioning_concatenate_mask,
            )
            
            # Process reference images for CLIP embedding if provided
            if reference_images is not None and hasattr(pipeline, 'image_encoder'):
                image_embeds_list = []
                
                for image in reference_images:
                    # Convert PIL image to tensor if needed
                    if not isinstance(image, torch.Tensor):
                        if hasattr(pipeline, 'image_processor'):
                            image = pipeline.image_processor(images=image, return_tensors="pt").to(device)
                        else:
                            # Fallback if no image_processor
                            from PIL import Image
                            if isinstance(image, Image.Image):
                                from torchvision import transforms
                                transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                                        std=[0.26862954, 0.26130258, 0.27577711])
                                ])
                                image = transform(image).unsqueeze(0).to(device)
                    
                    # Get visual embedding from the penultimate layer
                    with torch.no_grad():
                        image_embeds = pipeline.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                    
                    image_embeds_list.append(image_embeds)
                
                # Concatenate all reference embeddings
                all_image_embeds = torch.cat(image_embeds_list, dim=1)

        generation_kwargs = {
            "latents": latents,
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)

        from finetrainers.patches.dependencies.diffusers.control import \
            control_channel_concat
        
        def _get_model_input(*args, **kwargs):
            # Original model input function
            original_output = original_func(*args, **kwargs)
            
            # Add reference image embeddings to encoder_hidden_states if we have them
            if 'reference_images' in locals() and reference_images is not None:
                encoder_hidden_states = original_output[1]  # Assuming encoder_hidden_states is the second return value
                encoder_hidden_states = torch.cat([all_image_embeds, encoder_hidden_states], dim=1)
                
                # Replace encoder_hidden_states in the output
                outputs = list(original_output)
                outputs[1] = encoder_hidden_states
                return tuple(outputs)
            
            return original_output
        
        # Only patch if we have reference images
        if reference_images is not None and hasattr(pipeline, 'image_encoder'):
            # Store original method
            original_func = pipeline._encode_prompt
            
            # Temporarily patch the method
            pipeline._encode_prompt = _get_model_input
        
        try:
            with control_channel_concat(pipeline.transformer, ["hidden_states"], [control_latents], dims=[1]):
                video = pipeline(**generation_kwargs).frames[0]
        finally:
            # Restore original method if we patched it
            if reference_images is not None and hasattr(pipeline, 'image_encoder'):
                pipeline._encode_prompt = original_func

        return [VideoArtifact(value=video)]