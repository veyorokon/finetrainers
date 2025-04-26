from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel
from diffusers.utils import load_image
from transformers import CLIPImageProcessor, CLIPVisionModel

from finetrainers.data import VideoArtifact, initialize_reference_dataset
from finetrainers.logging import get_logger
from finetrainers.models.wan.reference_specification import WanReferenceModelSpecification
from finetrainers.trainer.control_trainer.trainer import ControlTrainer
from finetrainers.utils import get_non_null_items

from .config import ReferenceConfig, ReferenceType
from .data import IterableReferenceDataset, ValidationReferenceDataset

logger = get_logger()


class ReferenceTrainer(ControlTrainer):
    """Trainer for reference-based conditioning (A2-style).
    
    Extends ControlTrainer to add CLIP vision model processing for reference images.
    """
    
    def __init__(
        self,
        config: ReferenceConfig,
        model_specification: Optional[WanReferenceModelSpecification] = None,
        **kwargs,
    ) -> None:
        """Initialize the reference trainer.
        
        Args:
            config: Configuration for reference-based training
            model_specification: Specification for the model architecture
        """
        super().__init__(config, model_specification, **kwargs)
        
        # Cast config to the right type
        self.config = config
        
        # Additional model components for reference conditioning
        self.image_encoder = None
        self.image_processor = None
        
        logger.info(
            f"Initialized ReferenceTrainer with:\n"
            f"  Reference Type: {self.config.reference_type}\n"
            f"  VAE Resolution: {self.config.vae_resolution}\n"
            f"  CLIP Resolution: {self.config.clip_resolution}\n"
            f"  Reference Order: {self.config.reference_order}\n"
            f"  Repeat Frames: {self.config.repeat_frames}\n"
            f"  Reference Suffixes: {self.config.reference_suffixes}"
        )
    
    def _load_models(self) -> None:
        """Load all models required for training."""
        # First load the models from the parent class
        super()._load_models()
        
        # Additionally load the CLIP vision models
        if isinstance(self.model_specification, WanReferenceModelSpecification):
            embedding_models = self.model_specification.load_embedding_models()
            self.image_processor = embedding_models.get("image_processor")
            self.image_encoder = embedding_models.get("image_encoder")
            
            # Move models to the right device
            if self.image_encoder is not None:
                self.image_encoder.to(self.device)
            
            logger.info(f"Loaded CLIP vision models for reference conditioning")
        else:
            logger.warning(
                f"Model specification {type(self.model_specification).__name__} is not a "
                f"WanReferenceModelSpecification. CLIP vision models not loaded."
            )
    
    def _create_dataset(self) -> torch.utils.data.IterableDataset:
        """Create the dataset for training."""
        # Initialize the base reference dataset
        dataset = initialize_reference_dataset(
            self.config.data_root,
            reference_suffixes=self.config.reference_suffixes,
            dataset_type=self.config.dataset_type,
            infinite=True
        )
        
        # Create the iterable dataset with reference processing
        reference_config = {
            "vae_resolution": self.config.vae_resolution,
            "clip_resolution": self.config.clip_resolution,
            "reference_order": self.config.reference_order,
            "repeat_frames": self.config.repeat_frames
        }
        
        return IterableReferenceDataset(
            dataset,
            self.config.control_type,
            reference_config=reference_config,
            device=self.device
        )
    
    def _create_validation_dataset(self) -> Optional[torch.utils.data.IterableDataset]:
        """Create the validation dataset."""
        if self.config.validation_filename is None:
            return None
        
        # For now, just use the same validation dataset from the parent class
        # In the future, this could be extended to handle reference-specific validation
        validation_dataset = super()._create_validation_dataset()
        
        if validation_dataset is not None:
            reference_config = {
                "vae_resolution": self.config.vae_resolution,
                "clip_resolution": self.config.clip_resolution,
                "reference_order": self.config.reference_order,
                "repeat_frames": self.config.repeat_frames
            }
            
            return ValidationReferenceDataset(
                validation_dataset,
                self.config.control_type,
                reference_config=reference_config,
                device=self.device
            )
        
        return None
    
    def _encode_references(self, references):
        """Encode reference images with CLIP vision model."""
        if self.image_encoder is None or self.image_processor is None:
            logger.warning("CLIP vision models not loaded, skipping reference encoding")
            return None
        
        clip_images = []
        for ref_image in references:
            clip_images.append(ref_image)
        
        # Use the model specification to encode the images
        if isinstance(self.model_specification, WanReferenceModelSpecification):
            embedding_conditions = self.model_specification.prepare_embeddings(
                self.image_processor,
                self.image_encoder,
                clip_images
            )
            return embedding_conditions
        
        return None
    
    def _training_step(self, batch):
        """Perform a single training step."""
        latent_model_conditions = {}
        condition_model_conditions = {}
        embedding_model_conditions = {}
        
        # Extract conditions as in parent class
        if "caption" in batch:
            condition_model_conditions = self.model_specification.prepare_conditions(
                self.tokenizer, self.text_encoder, batch["caption"]
            )
        
        # Handle control images/videos for VAE encoding
        has_control_image = "control_image" in batch
        has_control_video = "control_video" in batch
        
        if has_control_image or has_control_video:
            latent_model_conditions = self.model_specification.prepare_latents(
                self.vae,
                image=batch.get("image"),
                video=batch.get("video"),
                control_image=batch.get("control_image"),
                control_video=batch.get("control_video"),
                generator=self.generator,
            )
        
        # Handle reference images for CLIP encoding
        if "clip_references" in batch:
            embedding_model_conditions = self._encode_references(batch["clip_references"])
        
        # Sample random sigmas for flow matching
        batch_size = condition_model_conditions["encoder_hidden_states"].shape[0]
        sigmas = self._get_sigmas(batch_size)
        
        # Forward pass through the model
        pred, target, sigmas = self.model_specification.forward(
            self.transformer,
            condition_model_conditions,
            latent_model_conditions,
            embedding_model_conditions,
            sigmas,
            generator=self.generator,
        )
        
        # Compute loss
        flow_matching_loss = F.mse_loss(pred, target, reduction="none")
        flow_matching_loss = flow_matching_loss.mean(dim=[1, 2, 3, 4])
        flow_matching_loss = (flow_matching_loss * sigmas).mean()
        
        return flow_matching_loss
    
    def validation(self):
        """Run validation."""
        if not self.do_validation:
            return
            
        # Use parent validation but add reference handling
        super().validation()