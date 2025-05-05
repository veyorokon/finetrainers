import json
import pathlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import datasets.distributed
import torch
import torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel
from diffusers.utils import load_image
from peft import get_peft_model_state_dict
from transformers import CLIPImageProcessor, CLIPVisionModel

from finetrainers import data, utils
from finetrainers.config import TrainingType
from finetrainers.data.reference import (generate_video_resolution_buckets,
                                         initialize_reference_dataset)
from finetrainers.logging import get_logger
from finetrainers.models.wan.reference_specification import \
    WanReferenceModelSpecification
from finetrainers.trainer.control_trainer.trainer import ControlTrainer

from .config import ReferenceConfig
from .data import IterableReferenceDataset

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
    
    def _prepare_checkpointing(self) -> None:
        parallel_backend = self.state.parallel_backend

        def save_model_hook(state_dict: Dict[str, Any]) -> None:
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if self.args.training_type == TrainingType.REFERENCE_LORA:
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    qk_norm_state_dict = None
                    if self.args.train_qk_norm:
                        qk_norm_state_dict = {
                            name: parameter
                            for name, parameter in state_dict.items()
                            if any(
                                re.search(identifier, name) is not None
                                for identifier in self.model_specification._qk_norm_identifiers
                            )
                            and parameter.numel() > 0
                        }
                        if len(qk_norm_state_dict) == 0:
                            qk_norm_state_dict = None
                    # fmt: off
                    metadata = {
                        "r": self.args.rank,
                        "lora_alpha": self.args.lora_alpha,
                        "init_lora_weights": True,
                        "target_modules": self._get_lora_target_modules(),
                        "rank_pattern": {self.model_specification.control_injection_layer_name: self.model_specification._original_control_layer_out_features},
                        "alpha_pattern": {self.model_specification.control_injection_layer_name: self.model_specification._original_control_layer_out_features},
                    }
                    metadata = {"lora_config": json.dumps(metadata, indent=4)}
                    # fmt: on
                    self.model_specification._save_lora_weights(
                        self.args.output_dir, state_dict, qk_norm_state_dict, self.scheduler, metadata
                    )
                elif self.args.training_type == TrainingType.REFERENCE_FULL_FINETUNE:
                    self.model_specification._save_model(
                        self.args.output_dir, self.transformer, state_dict, self.scheduler
                    )
            parallel_backend.wait_for_everyone()

        enable_state_checkpointing = self.args.checkpointing_steps > 0
        self.checkpointer = parallel_backend.get_checkpointer(
            dataloader=self.dataloader,
            model_parts=[self.transformer],
            optimizers=self.optimizer,
            schedulers=self.lr_scheduler,
            states={"train_state": self.state.train_state},
            checkpointing_steps=self.args.checkpointing_steps,
            checkpointing_limit=self.args.checkpointing_limit,
            output_dir=self.args.output_dir,
            enable=enable_state_checkpointing,
            _callback_fn=save_model_hook,
        )

        resume_from_checkpoint = self.args.resume_from_checkpoint
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = -1
        if resume_from_checkpoint is not None:
            self.checkpointer.load(resume_from_checkpoint)
    
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
            "reference_suffixes": self.config.reference_suffixes,
            "vae_combine": self.config.vae_combine
        }
        
        logger.info(f"Using vae_combine method: {self.config.vae_combine}")
        
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
            
            # vae_combine should only exist in reference_config
            if "vae_combine" in config:
                config.pop("vae_combine")
            
            # Update the trainer config with values from the dataset config
            if reference_config:
                logger.info(f"Found reference_config in dataset: {reference_config}")
                if "repeat_frames" in reference_config:
                    self.config.repeat_frames = reference_config["repeat_frames"]
                    logger.info(f"Updated repeat_frames from dataset config: {self.config.repeat_frames}")
                if "vae_resolution" in reference_config:
                    self.config.vae_resolution = reference_config["vae_resolution"]
                if "clip_resolution" in reference_config:
                    self.config.clip_resolution = reference_config["clip_resolution"]
                if "reference_order" in reference_config:
                    self.config.reference_order = reference_config["reference_order"]
                if "vae_combine" in reference_config:
                    self.config.vae_combine = reference_config["vae_combine"]
                    logger.info(f"Updated vae_combine from reference_config: {self.config.vae_combine}")
            
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
            "reference_suffixes": self.config.reference_suffixes,
            "vae_combine": self.config.vae_combine
        }
        
        logger.info(f"Using vae_combine method: {self.config.vae_combine}")
        
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
        
    def create_validation_dataset(self, validation_file: str, local_rank: int, dp_world_size: int) -> torch.utils.data.IterableDataset:
        """Create a validation dataset for the reference trainer.
        
        This method is called by the parent ControlTrainer._validate method.
        
        Args:
            validation_file: The path to the validation file
            local_rank: The local rank for distributed training
            dp_world_size: The world size for distributed training
            
        Returns:
            A validation dataset
        """
        logger.info(f"Creating dataset from {validation_file} using IterableReferenceDataset")
        
        # Load JSON data with field="data" parameter to extract examples directly
        filename = pathlib.Path(validation_file)
        raw_data = datasets.load_dataset("json", data_files=filename.as_posix(), field="data", split="train")
        iterable_data = raw_data.to_iterable_dataset()
        split_data = datasets.distributed.split_dataset_by_node(iterable_data, local_rank, dp_world_size)
        
        # Create reference config 
        reference_config = {
            "vae_resolution": self.config.vae_resolution,
            "clip_resolution": self.config.clip_resolution,
            "reference_order": self.config.reference_order,
            "repeat_frames": self.config.repeat_frames,
            "reference_suffixes": self.config.reference_suffixes,
            "vae_combine": self.config.vae_combine
        }
        
        # Use the same IterableReferenceDataset as training, for consistent processing
        return IterableReferenceDataset(
            split_data,
            self.config.control_type,
            reference_config=reference_config,
            device=self.state.parallel_backend.device
        )
    
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
        
    def validation(self):
        """Run validation."""
        if not self.do_validation:
            return
            
        # Use parent validation but add reference handling
        super().validation()