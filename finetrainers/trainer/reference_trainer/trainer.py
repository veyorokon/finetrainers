import json
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import datasets.distributed
import torch
import torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel
from diffusers.utils import load_image
from transformers import CLIPImageProcessor, CLIPVisionModel

from finetrainers import data
from finetrainers.data import PatternReferenceDataset, VideoArtifact
from finetrainers.data.reference import initialize_reference_dataset
from finetrainers.logging import get_logger
from finetrainers.models.wan.reference_specification import \
    WanReferenceModelSpecification
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
    
    def _prepare_models(self) -> None:
        """Override parent method to use the correct in_channels value.
        
        Instead of reusing the parent method which doubles the in_channels, 
        we implement our own version that uses the correct value directly.
        """
        logger.info("Initializing models for reference training")

        # Get the in_channels directly from the transformer config (36 for Wan2)
        in_channels = self.model_specification.transformer_config.in_channels
        logger.info(f"Using in_channels value: {in_channels} from model config")
        
        # Load diffusion components with the exact in_channels we want
        diffusion_components = self.model_specification.load_diffusion_models(in_channels)
        self._set_components(diffusion_components)
        
        if self.state.parallel_backend.pipeline_parallel_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not supported yet. This will be supported in the future."
            )

    def _load_models(self) -> None:
        """Load all models required for training."""
        # Call our overridden _prepare_models instead of the parent's
        self._prepare_models()
        
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
            "repeat_frames": self.config.repeat_frames,
            "reference_suffixes": self.config.reference_suffixes
        }
        
        # Also pass same reference config to the model specification
        if isinstance(self.model_specification, WanReferenceModelSpecification):
            self.model_specification.reference_config = reference_config
        
        return IterableReferenceDataset(
            dataset,
            self.config.control_type,
            reference_config=reference_config,
            device=self.device
        )
        
    def _prepare_dataset(self) -> None:
        """Override parent method to use IterableReferenceDataset instead of IterableControlDataset."""
        logger.info("Initializing reference dataset and dataloader")

        with open(self.args.dataset_config, "r") as file:
            dataset_configs = json.load(file)["datasets"]
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")

        datasets = []
        for config in dataset_configs:
            data_root = config.pop("data_root", None)
            dataset_file = config.pop("dataset_file", None)
            dataset_type = config.pop("dataset_type")
            caption_options = config.pop("caption_options", {})
            reference_suffixes = config.pop("reference_suffixes", ["_object", "_background"])
            reference_config = config.pop("reference_config", {})
            
            # Handle auto-generating video_resolution_buckets if needed
            if "video_resolutions" in config and "video_resolution_buckets" not in config and data_root:
                logger.info(f"Auto-generating video_resolution_buckets from {data_root}")
                video_resolutions = config.pop("video_resolutions")
                video_resolution_buckets = generate_video_resolution_buckets(
                    data_root,
                    video_resolutions,
                    reference_suffixes
                )
                config["video_resolution_buckets"] = video_resolution_buckets
                logger.info(f"Generated {len(video_resolution_buckets)} video_resolution_buckets")
            elif "video_resolutions" in config and data_root is None:
                logger.warning("Cannot auto-generate video_resolution_buckets without data_root")

            if data_root is not None and dataset_file is not None:
                raise ValueError("Both data_root and dataset_file cannot be provided in the same dataset config.")

            dataset_name_or_root = data_root or dataset_file
            
            # Use the appropriate dataset initialization based on type
            if dataset_type == "video_references":
                logger.info(f"Initializing reference dataset from {dataset_name_or_root}")
                dataset = initialize_reference_dataset(
                    dataset_name_or_root, 
                    reference_suffixes=reference_suffixes,
                    dataset_type=dataset_type, 
                    infinite=True
                )
            else:
                dataset = data.initialize_dataset(
                    dataset_name_or_root, dataset_type, streaming=True, infinite=True, _caption_options=caption_options
                )

            if not dataset._precomputable_once and self.args.precomputation_once:
                raise ValueError(
                    f"Dataset {dataset_name_or_root} does not support precomputing all embeddings at once."
                )

            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
            datasets.append(dataset)

        dataset = data.combine_datasets(datasets, buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True)
        
        # Use IterableReferenceDataset instead of IterableControlDataset
        reference_config = {
            "vae_resolution": self.config.vae_resolution,
            "clip_resolution": self.config.clip_resolution,
            "reference_order": self.config.reference_order, 
            "repeat_frames": self.config.repeat_frames,
            "reference_suffixes": self.config.reference_suffixes
        }
        
        logger.info(f"Creating IterableReferenceDataset with config: {reference_config}")
        dataset = IterableReferenceDataset(
            dataset, 
            self.config.control_type, 
            reference_config=reference_config,
            device=self.state.parallel_backend.device
        )
        
        # Define a custom collate function to handle PIL images with any batch size
        def reference_collate_fn(batch):
            if len(batch) == 0:
                return {}
                
            # For batch size 1, just return the first item directly
            if len(batch) == 1:
                return batch[0]
                
            # For larger batch sizes, we need to handle each key appropriately
            result = {}
            elem = batch[0]
            
            for key in elem:
                if key == 'vae_references':
                    # Each sample has its own references - keep as list of lists
                    result[key] = [b[key] for b in batch]
                elif key == 'clip_references':
                    # Each sample has its own clip references - keep as list of lists
                    result[key] = [b[key] for b in batch]
                elif key == 'references':
                    # Each sample has its own references dict - keep as list of dicts
                    result[key] = [b[key] for b in batch]
                else:
                    # For standard tensor data, use standard batching
                    values = [b[key] for b in batch]
                    if isinstance(values[0], torch.Tensor):
                        result[key] = torch.stack(values)
                    elif isinstance(values[0], (int, float, str, bool)):
                        result[key] = values
                    else:
                        # For other types, keep as list
                        result[key] = values
            
            return result
            
        # Use the custom collate function
        collate_fn = reference_collate_fn
        
        logger.info("Using custom collate_fn for reference dataset")
        dataloader = self.state.parallel_backend.prepare_dataloader(
            dataset, batch_size=1, num_workers=self.args.dataloader_num_workers, 
            pin_memory=self.args.pin_memory, collate_fn=collate_fn
        )

        self.dataset = dataset
        self.dataloader = dataloader
    
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
                "repeat_frames": self.config.repeat_frames,
                "reference_suffixes": self.config.reference_suffixes
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
        
        # Convert 'images' key to 'references' for our processor
        if "images" in batch and "references" not in batch:
            logger.info(f"Converting 'images' key to 'references' for processor compatibility")
            # Check if images is a list of paths or already a dictionary
            if isinstance(batch["images"], list):
                # Convert list of paths to dictionary with object/background keys
                # based on filenames
                reference_dict = {}
                for image_path in batch["images"]:
                    # Try to extract reference type from the filename
                    path = pathlib.Path(image_path)
                    filename = path.name
                    for suffix in self.config.reference_suffixes:
                        if suffix in filename:
                            # Strip leading underscore if present
                            ref_type = suffix[1:] if suffix.startswith("_") else suffix
                            reference_dict[ref_type] = image_path
                            break
                
                logger.info(f"Converted image paths to reference dictionary: {list(reference_dict.keys())}")
                batch["references"] = reference_dict
            else:
                logger.info("References are in the batch")
                # Images is already in the right format
                batch["references"] = batch["images"]
            
            # Keep images for other processing if needed
            
        # Extract conditions as in parent class
        if "caption" in batch:
            condition_model_conditions = self.model_specification.prepare_conditions(
                self.tokenizer, self.text_encoder, batch["caption"]
            )
        
        # Handle control images/videos for VAE encoding
        has_control_image = "control_image" in batch
        has_control_video = "control_video" in batch
        has_references = "vae_references" in batch and len(batch["vae_references"]) > 0
        has_refs = "references" in batch
        
        # Add detailed logging about what's in the batch
        logger.info(f"===== TRAINING STEP =====")
        logger.info(f"Training batch contains keys: {list(batch.keys())}")
        
        # Log vae references if present
        if has_references:
            logger.info(f"Found vae_references with {len(batch['vae_references'])} items")
            for i, ref in enumerate(batch["vae_references"]):
                img_type = type(ref["image"]).__name__
                repeat = ref["repeat"]
                img_info = f"size={ref['image'].size}" if hasattr(ref["image"], "size") else "no size"
                logger.info(f"  vae_reference {i}: type={img_type}, repeat={repeat}, {img_info}")
        else:
            logger.info("No vae_references found in batch")
            
        # Log raw references if present
        if has_refs:
            logger.info(f"Found references with keys: {list(batch['references'].keys()) if isinstance(batch['references'], dict) else 'list'}")
            if "images" in batch:
                logger.info(f"Converted from 'images' with {len(batch['images']) if isinstance(batch['images'], list) else 'unknown'} items")
        else:
            logger.info("No references found in batch")
            
        # Log control inputs
        if has_control_image:
            control_img = batch["control_image"]
            logger.info(f"Control image present: {type(control_img).__name__}, shape={control_img.shape if hasattr(control_img, 'shape') else 'no shape'}")
        
        if has_control_video:
            control_vid = batch["control_video"]
            logger.info(f"Control video present: {type(control_vid).__name__}, shape={control_vid.shape if hasattr(control_vid, 'shape') else 'no shape'}")
            
        # Log whether we have normal inputs
        if "image" in batch and batch["image"] is not None:
            logger.info(f"Image present with shape {batch['image'].shape}")
            
        if "video" in batch and batch["video"] is not None:
            logger.info(f"Video present with shape {batch['video'].shape}")
            
        # Pass everything in batch to prepare_latents when we have control inputs
        if has_control_image or has_control_video:
            logger.info("Calling prepare_latents with control inputs")
            latent_model_conditions = self.model_specification.prepare_latents(
                self.vae,
                image=batch.get("image"),
                video=batch.get("video"),
                control_image=batch.get("control_image"),
                control_video=batch.get("control_video"),
                generator=self.generator,
                # We're not adding vae_references here yet, just keeping the original flow
            )
            
        # Add logic for handling references - temporarily commented out
        # elif has_references:
        #    logger.info("Calling prepare_latents with references")
        #    latent_model_conditions = self.model_specification.prepare_latents(
        #        self.vae,
        #        image=batch.get("image"),
        #        video=batch.get("video"), 
        #        generator=self.generator,
        #        vae_references=batch["vae_references"]
        #    )
        
        # Handle reference images for CLIP encoding if available
        if "clip_references" in batch and len(batch["clip_references"]) > 0:
            ref_encoding = self._encode_references(batch["clip_references"])
            if ref_encoding is not None:
                embedding_model_conditions = ref_encoding
        
        # Sample random sigmas for flow matching
        batch_size = condition_model_conditions["encoder_hidden_states"].shape[0]
        sigmas = self._get_sigmas(batch_size)
        
        # Forward pass through the model
        pred, target, sigmas = self.model_specification.forward(
            self.transformer,
            condition_model_conditions,
            latent_model_conditions,
            sigmas,
            embedding_model_conditions=embedding_model_conditions,
            generator=self.generator,
        )
        
        # Compute loss
        flow_matching_loss = F.mse_loss(pred, target, reduction="none")
        flow_matching_loss = flow_matching_loss.mean(dim=[1, 2, 3, 4])
        flow_matching_loss = (flow_matching_loss * sigmas).mean()
        
        return flow_matching_loss
    
    # Since we now use a proper processor for reference-to-control conversion,
    # we don't need to override the _prepare_data method anymore.
    # The parent ControlTrainer's _prepare_data method works fine with our setup.
    # All reference processing happens in our ReferenceToControlProcessor during preprocessing
        
    def validation(self):
        """Run validation."""
        if not self.do_validation:
            return
            
        # Use parent validation but add reference handling
        super().validation()