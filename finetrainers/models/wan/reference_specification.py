from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import CLIPImageProcessor, CLIPVisionModel

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.processors import (ProcessorMixin, ReferenceClipProcessor,
                                     ReferenceToControlProcessor)
from finetrainers.typing import ArtifactType
from finetrainers.utils import get_non_null_items

from .base_specification import WanLatentEncodeProcessor
from .control_specification import WanControlModelSpecification

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
        # Require reference_config to be provided
        if reference_config is None:
            raise ValueError("WanReferenceModelSpecification requires reference_config")
        self.reference_config = reference_config
        logger.info(f"Initialized WanReferenceModelSpecification with reference_config: {self.reference_config}")
        
        # Create reference-to-control processor if not provided
        if control_model_processors is None:
            # Import the reference-specific encoder processor
            from finetrainers.processors.reference import \
                WanReferenceLatentEncodeProcessor

            # Create a reference processor and specialized latent processor
            reference_processor = ReferenceToControlProcessor(
                ["image", "video"], 
                reference_config=self.reference_config
            )
            
            reference_latent_processor = WanReferenceLatentEncodeProcessor(
                ["control_latents", "__drop__", "__drop__"]
            )
            
            # Add logging to debug more easily
            logger.info("Initializing reference control processors:")
            logger.info(f"  Reference processor output names: {reference_processor.output_names}")
            logger.info(f"  Reference latent processor output names: {reference_latent_processor.output_names}")
            logger.info(f"  Reference processor reference_config: {reference_processor.reference_config}")
            
            # Use both processors in sequence
            control_model_processors = [reference_processor, reference_latent_processor]
        
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
        from finetrainers.trainer.reference_trainer.data import \
            apply_reference_frame_conditioning

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
        
        # Diagnostic logging - check the mask channels in final concatenated tensor
        # Mask channels are now at positions 0-3 (before the content channels)
        logger.info(f"Checking mask channels (first 4 channels) in concatenated tensor:")
        for i in range(4):
            mask_channel = noisy_latents[:, i]
            mask_min = mask_channel.min().item()
            mask_max = mask_channel.max().item()
            mask_mean = mask_channel.mean().item()
            mask_nonzero = (mask_channel > 0.5).float().sum().item()
            logger.info(f"Mask channel {i} stats: min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}, " +
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
            
            # Save individual mask channels for quick reference
            # Mask channels are now at indices 0-3 in final tensor
            save_latent_channels(noisy_latents, output_dir, "mask", [0, 1, 2, 3])
            
            # Create channel×frame grid visualization
            # Group sizes now match revised A2 inference order: mask/padding (4), content (16), conditioning (16)
            create_channel_frame_grid(
                noisy_latents,
                output_dir,
                filename=f"latent_grid_{step_id}.png",
                group_sizes=[4, 16, 16]
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
        generator: Optional[torch.Generator] = None,
        prompt: Optional[str] = None,
        caption: Optional[str] = None,
        vae_references: Optional[List[Dict[str, Any]]] = None,
        clip_references: Optional[List[torch.Tensor]] = None,
        references: Optional[Dict[str, str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        **kwargs,
    ) -> List[ArtifactType]:
        """
        Run validation using our unified format.
        
        This method takes data in our unified dataset format (same as training) and
        performs generation using the pipeline.
        
        Args:
            pipeline: The WanPipeline instance for generation
            generator: Random number generator
            prompt/caption: Text prompt for generation (caption is alias for prompt)
            vae_references: Processed reference images with repeat counts
            clip_references: Reference images for CLIP embedding
            references: Dictionary mapping reference types to file paths
            height, width, num_frames: Output dimensions
            num_inference_steps: Number of denoising steps
            
        Returns:
            List of artifacts (typically a single VideoArtifact)
        """
        from finetrainers.data import VideoArtifact
        from finetrainers.processors.reference import ReferenceToControlProcessor
        from finetrainers.trainer.reference_trainer.data import apply_reference_frame_conditioning
        from finetrainers.patches.dependencies.diffusers.control import control_channel_concat

        logger.info(f"=== Starting validation with unified format ===")
        
        with torch.no_grad():
            # Use text prompt (try both prompt and caption fields)
            text_prompt = prompt if prompt is not None else caption
            if text_prompt is None:
                text_prompt = ""
                logger.warning("No prompt or caption provided for validation")
            
            logger.info(f"Using text prompt: '{text_prompt}'")
                
            # Process device and dtype
            device = pipeline._execution_device
            dtype = pipeline.vae.dtype
            
            # Prepare initial latents for video generation
            in_channels = self.transformer_config.in_channels  # Original in_channels (not doubled)
            latents = pipeline.prepare_latents(1, in_channels, height, width, num_frames, dtype, device, generator)
            
            # Prepare VAE scaling factors
            latents_mean = (
                torch.tensor(self.vae_config.latents_mean)
                .view(1, self.vae_config.z_dim, 1, 1, 1)
                .to(device, dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_config.latents_std).view(1, self.vae_config.z_dim, 1, 1, 1).to(
                device, dtype
            )
            
            # Process control inputs - try all possible sources
            control_video = None
            
            # If we have reference data, process it using same processors as training
            if (vae_references and len(vae_references) > 0) or (references and len(references) > 0):
                logger.info("Processing references for control conditioning")
                
                # Use same processor as training to maintain consistency
                reference_processor = ReferenceToControlProcessor(
                    ["image", "video"], 
                    reference_config=self.reference_config
                )
                
                # Process references to get control video
                result = reference_processor(
                    references=references,
                    vae_references=vae_references
                )
                
                # Get control video from processor result
                video_list = result.get("video")
                if video_list and len(video_list) > 0:
                    # Take first video for now (same as training)
                    control_video = video_list[0]
                    logger.info(f"Created control video with shape {control_video.shape}")
            
            # If we don't have control input yet, check if it was provided directly
            if control_video is None and "control_video" in kwargs:
                control_video = kwargs["control_video"]
                logger.info(f"Using provided control_video with shape {control_video.shape}")
            elif control_video is None and "control_image" in kwargs:
                # Convert control image to video format
                control_image = kwargs["control_image"]
                logger.info(f"Converting control_image to video format")
                control_video = pipeline.video_processor.preprocess(
                    control_image, height=height, width=width
                ).unsqueeze(2)
            
            # If we still don't have control input, we can't continue
            if control_video is None:
                raise ValueError("No control inputs available for validation. Provide references, control_video, or control_image.")
                
            # Convert to latents
            control_video = control_video.to(device=device, dtype=dtype)
            control_latents = pipeline.vae.encode(control_video).latent_dist.mode()
            control_latents = self._normalize_latents(control_latents, latents_mean, latents_std)
            
            # Apply reference frame conditioning (same as in training)
            control_latents = apply_reference_frame_conditioning(
                control_latents,
                latents.shape[2],
                frame_conditioning_type=self.frame_conditioning_type or "full",
                frame_conditioning_index=self.frame_conditioning_index or 0,
                channel_dim=1,
                frame_dim=2,
                concatenate_mask=self.frame_conditioning_concatenate_mask,
            )
            
            # Process CLIP reference images for embedding
            clip_embeddings = None
            if clip_references and len(clip_references) > 0 and hasattr(pipeline, 'image_encoder'):
                logger.info(f"Processing {len(clip_references)} reference images for CLIP embedding")
                
                image_embeds_list = []
                for image in clip_references:
                    # Process image with CLIP processor
                    if hasattr(pipeline, 'image_processor'):
                        image = pipeline.image_processor(images=image, return_tensors="pt").to(device)
                    
                    # Get embedding from CLIP vision model
                    with torch.no_grad():
                        image_embeds = pipeline.image_encoder(image, output_hidden_states=True).hidden_states[-2]
                    
                    image_embeds_list.append(image_embeds)
                
                # Concatenate all reference embeddings
                if image_embeds_list:
                    clip_embeddings = torch.cat(image_embeds_list, dim=1)
                    logger.info(f"Created CLIP embeddings with shape {clip_embeddings.shape}")
            
            # Create function to patch pipeline's _encode_prompt
            original_func = None
            if clip_embeddings is not None:
                logger.info("Patching pipeline's _encode_prompt to include CLIP embeddings")
                
                def _patched_encode_prompt(*args, **kwargs):
                    # Call original function
                    original_output = original_func(*args, **kwargs)
                    
                    # Add reference image embeddings to encoder_hidden_states
                    encoder_hidden_states = original_output[1]  # encoder_hidden_states is the second return
                    encoder_hidden_states = torch.cat([clip_embeddings, encoder_hidden_states], dim=1)
                    
                    # Replace in the output
                    outputs = list(original_output)
                    outputs[1] = encoder_hidden_states
                    return tuple(outputs)
                
                # Store original method for later restoration
                original_func = pipeline._encode_prompt
                pipeline._encode_prompt = _patched_encode_prompt
            
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": text_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "return_dict": True,
                "output_type": "pt",  # Return raw tensors for better control
            }
            
            # Remove None values
            generation_kwargs = get_non_null_items(generation_kwargs)
            
            try:
                # Generate with patched pipeline and control latents
                with control_channel_concat(pipeline.transformer, ["hidden_states"], [control_latents], dims=[1]):
                    logger.info(f"Running pipeline generation with parameters: {generation_kwargs.keys()}")
                    result = pipeline(**generation_kwargs)
                    video = result.frames[0]
            finally:
                # Restore original method if we patched it
                if original_func is not None:
                    pipeline._encode_prompt = original_func
            
            # Return as VideoArtifact (same as parent class)
            return [VideoArtifact(value=video)]