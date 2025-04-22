"""E2V (Elements-to-Video) Trainer implementation.

This module implements the E2V trainer, which handles training Wan models
using multiple reference images (elements) as conditioning for video generation.

The trainer follows the separation of concerns pattern where:
1. Preprocessing is done in the dataset layer
2. Model inference is done in the trainer layer
3. Configuration is strictly driven by the config file
"""
import functools
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

import datasets.distributed
import torch
import torch.backends
import wandb
from diffusers import DiffusionPipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict
from tqdm import tqdm

from finetrainers import data, logging, optimizer, parallel, patches, utils
from finetrainers.data import DPDataLoader
from finetrainers.config import TrainingType
from finetrainers.logging import get_logger
from finetrainers.patches import load_lora_weights
from finetrainers.state import State, TrainState

from .config import E2VFullRankConfig, E2VLowRankConfig
from .data import IterableE2VDataset, ValidationE2VDataset
from .encoders import ENCODER_REGISTRY, encode_vae, encode_clip
from .combiners import COMBINER_REGISTRY, combine_vae_features, combine_clip_features
from .utils import validate_e2v_config, validate_tensor_combinations, find_tensor_by_key_pattern

logger = get_logger()


class E2VTrainer:
    """Trainer for Elements-to-Video (E2V) models.
    
    Handles the full training lifecycle including:
    - Model initialization and setup
    - Dataset preparation
    - Training loop management
    - Validation and visualization
    - Checkpointing and logging
    
    The E2V approach uses multiple reference images to condition video generation,
    processing them through both VAE (spatial) and CLIP (semantic) pathways.
    """
    
    # Component lists for organized handling
    _all_component_names = [
        "tokenizer", "tokenizer_2", "tokenizer_3", 
        "text_encoder", "text_encoder_2", "text_encoder_3", 
        "transformer", "unet", "vae", "scheduler", "image_encoder"
    ]
    _condition_component_names = [
        "tokenizer", "tokenizer_2", "tokenizer_3", 
        "text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"
    ]
    _latent_component_names = ["vae"]
    _diffusion_component_names = ["transformer", "unet", "scheduler"]

    def __init__(self, args: Union[E2VFullRankConfig, E2VLowRankConfig], model_specification):
        """Initialize the E2V trainer.
        
        Args:
            args: Configuration containing training parameters
            model_specification: Model specification defining architecture
        """
        self.args = args
        self.state = State()
        self.state.train_state = TrainState()
        
        # Initialize components
        self._init_component_attributes()
        
        # Initialize training environment
        self._init_distributed()
        self._init_config_options()
        self._init_logging()
        self._init_directories_and_repositories()
        
        # Set up model specification
        patches.perform_patches_for_training(self.args, self.state.parallel_backend)
        self.model_specification = model_specification
        self._are_condition_models_loaded = False
        
        # Pass frame conditioning parameters to model specification
        self._init_frame_conditioning()

    def _init_component_attributes(self):
        """Initialize all model component attributes to None."""
        # Tokenizers
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None
        
        # Encoders
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        self.image_encoder = None
        
        # Generation components
        self.transformer = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        
        # Training components
        self.optimizer = None
        self.lr_scheduler = None
        self.checkpointer = None
        
        # Track parameters
        self.state.num_trainable_parameters = 0

    def _init_distributed(self):
        """Initialize distributed training backend."""
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

        backend_cls = parallel.get_parallel_backend_cls(self.args.parallel_backend)
        self.state.parallel_backend = backend_cls(
            world_size=world_size,
            pp_degree=self.args.pp_degree,
            dp_degree=self.args.dp_degree,
            dp_shards=self.args.dp_shards,
            cp_degree=self.args.cp_degree,
            tp_degree=self.args.tp_degree,
            backend="nccl",
            timeout=self.args.init_timeout,
            logging_dir=self.args.logging_dir,
            output_dir=self.args.output_dir,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        if self.args.seed is not None:
            self.state.parallel_backend.enable_determinism(self.args.seed)

    def _init_config_options(self):
        """Initialize configuration options and system settings."""
        # Gradient accumulation
        self.state.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        
        # Logging and precision options
        self.state.logging_nan_or_inf = getattr(self.args, "logging_nan_or_inf", False)
        self.state.allow_tf32 = self.args.allow_tf32
        if self.state.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Batch sizes
        if not hasattr(self.args, "train_batch_size") or not self.args.train_batch_size:
            self.args.train_batch_size = 1
        
        if not hasattr(self.args, "eval_batch_size") or not self.args.eval_batch_size:
            self.args.eval_batch_size = 1
        
        # Torch compilation settings
        if hasattr(self.args, "setup_torch_compile") and self.args.setup_torch_compile:
            os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

    def _init_frame_conditioning(self):
        """Initialize frame conditioning parameters for model specification."""
        # Get frame conditioning params with defaults
        frame_conditioning_type = getattr(self.args, "frame_conditioning_type", "full")
        frame_conditioning_index = getattr(self.args, "frame_conditioning_index", 0)
        frame_conditioning_concatenate_mask = getattr(self.args, "frame_conditioning_concatenate_mask", True)
        
        # Initialize model specification with these parameters
        self.model_specification._trainer_init(
            frame_conditioning_type, frame_conditioning_index, frame_conditioning_concatenate_mask
        )

    def _init_logging(self):
        """Initialize logging functionality."""
        # Only log from main process
        if self.state.parallel_backend.is_main_process:
            logger.info(f"E2V training: {self.args.training_type}")
            logger.info(f"Output directory: {self.args.output_dir}")
            
            # Log key configuration parameters
            logger.info(f"Training batch size: {self.args.train_batch_size}")
            logger.info(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
            
            # Log LoRA parameters if applicable
            if self.args.training_type == TrainingType.E2V_LORA:
                logger.info(f"LoRA rank: {self.args.rank}")
                logger.info(f"LoRA alpha: {self.args.lora_alpha}")
                logger.info(f"LoRA target modules: {self.args.target_modules}")

    def _init_trackers(self):
        """Initialize training trackers (e.g., Weights & Biases)."""
        parallel_backend = self.state.parallel_backend
        
        # Configure trackers following framework pattern
        trackers = [self.args.report_to]
        experiment_name = getattr(self.args, "tracker_name", None) or "finetrainers-experiment"
        parallel_backend.initialize_trackers(
            trackers, 
            experiment_name=experiment_name, 
            config=self._get_training_info(), 
            log_dir=self.args.logging_dir
        )

    def _get_training_info(self):
        """Get training information for logging to trackers."""
        info = self.args.to_dict()
        
        # Filter out irrelevant diffusion arguments
        diffusion_args = info.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__ if self.scheduler is not None else ""
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args
        
        info.update({"diffusion_arguments": filtered_diffusion_args})
        return info

    def _init_directories_and_repositories(self):
        """Initialize output directories and HF Hub repositories."""
        if self.state.parallel_backend.is_main_process:
            # Ensure output directory exists
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = Path(self.args.output_dir)
            
            # Initialize Hub repository if configured
            if self.args.push_to_hub:
                repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
                self.state.repo_id = create_repo(
                    token=self.args.hub_token, 
                    repo_id=repo_id, 
                    exist_ok=True
                ).repo_id

    def run(self):
        """Main entry point for training - follows framework pattern."""
        parallel_backend = self.state.parallel_backend
        start_time = time.time()
        
        try:
            # Step 1: Initialize and prepare all model components
            logger.info("Starting E2V training initialization")
            self._prepare_models()
            
            # Step 2: Set up trainable parameters (LoRA or full)
            self._prepare_trainable_parameters()
            
            # Step 3: Initialize optimizer and scheduler
            self._prepare_for_training()
            
            # Step 4: Load and prepare dataset
            self._prepare_dataset()
            
            # Step 5: Set up checkpointing
            self._prepare_checkpointing()
            
            # Step 6: Run training loop
            logger.info("Starting E2V training")
            self._train()
            
            # Log training time
            total_time = time.time() - start_time
            logger.info(f"E2V training completed in {total_time:.2f} seconds")
            
            # Final validation on training completion if requested
            if hasattr(self, "validation_dataloader") and self.validation_dataloader is not None and \
               hasattr(self.args, "run_validation_on_train_end") and self.args.run_validation_on_train_end:
                logger.info("Running final validation")
                self._validate(step=self.state.train_state.step, final_validation=True)
                
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            # Save checkpoint on interrupt if requested
            if hasattr(self.args, "save_on_interrupt") and self.args.save_on_interrupt and hasattr(self, "checkpointer"):
                logger.info("Saving checkpoint on interrupt")
                if hasattr(self, "checkpointer") and self.checkpointer is not None:
                    self.checkpointer.save()
            raise
            
        except Exception as e:
            logger.exception(f"Error during E2V training: {e}")
            # Always re-raise the exception for proper error reporting
            raise
            
        finally:
            # Cleanup resources regardless of success/failure
            self._cleanup()
            
    def _cleanup(self):
        """Clean up resources after training."""
        logger.info("Cleaning up resources")
        
        # 1. Clean up trackers
        try:
            self.state.parallel_backend.cleanup_trackers()
        except Exception as e:
            logger.warning(f"Error cleaning up trackers: {e}")
        
        # 2. Free up GPU memory
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"Error cleaning GPU memory: {e}")
            
        # 3. Destroy process group for distributed training
        if self.state.parallel_backend is not None:
            try:
                self.state.parallel_backend.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up parallel backend: {e}")
            
    def _prepare_models(self):
        """Prepare models for training following framework patterns."""
        parallel_backend = self.state.parallel_backend
        logger.info("Preparing models")
        
        # 1. Load models in the correct order
        self._load_condition_models()
        self._load_latent_models()
        self._load_diffusion_models()
        
        # 2. Move models to appropriate devices
        self._move_components_to_device()
        
        # 3. Apply activation checkpointing if configured
        if self.args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing")
            utils.apply_activation_checkpointing(self.transformer, checkpointing_type="full")
            
        # 4. Apply compile if specified
        if "transformer" in self.args.compile_modules:
            logger.info("Compiling transformer model")
            utils.apply_compile(self.transformer)
            
        # 5. Apply tensor parallelism if enabled
        if parallel_backend.tensor_parallel_enabled:
            logger.info("Applying tensor parallelism")
            self.model_specification.apply_tensor_parallel(
                backend=parallel.ParallelBackendEnum.PTD,
                device_mesh=parallel_backend.get_mesh("tp"),
                transformer=self.transformer,
            )
            
        # 6. Apply distributed data parallelism or sharding if needed
        self._apply_distributed_strategy()
            
    def _apply_distributed_strategy(self):
        """Apply appropriate distributed training strategy."""
        parallel_backend = self.state.parallel_backend
        
        if parallel_backend.data_sharding_enabled:
            # Handle FSDP or HSDP
            if parallel_backend.data_replication_enabled:
                logger.info("Applying HSDP to the model")
            else:
                logger.info("Applying FSDP to the model")
                
            # Apply FSDP or HSDP
            if parallel_backend.data_replication_enabled or parallel_backend.context_parallel_enabled:
                dp_mesh_names = ("dp_replicate", "dp_shard_cp")
            else:
                dp_mesh_names = ("dp_shard_cp",)
                
            parallel_backend.apply_fsdp2(
                model=self.transformer,
                dp_mesh=parallel_backend.get_mesh()[dp_mesh_names],
                param_dtype=self.args.transformer_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                pp_enabled=parallel_backend.pipeline_parallel_enabled,
                cpu_offload=False,
            )
        elif parallel_backend.data_replication_enabled:
            logger.info("Applying DDP to the model")
            
            if parallel_backend.get_mesh().ndim > 1:
                raise ValueError("DDP not supported for > 1D parallelism")
                
            parallel_backend.apply_ddp(self.transformer, parallel_backend.get_mesh())
        else:
            parallel_backend.prepare_model(self.transformer)
            
    def _move_components_to_device(self):
        """Move all model components to the appropriate device."""
        device = self.state.parallel_backend.device
        logger.info(f"Moving model components to device: {device}")
        
        # Move condition components
        for component_name in self._condition_component_names:
            component = getattr(self, component_name, None)
            if component is not None and hasattr(component, "to"):
                component = component.to(device)
                setattr(self, component_name, component)
                
        # Move latent components
        for component_name in self._latent_component_names:
            component = getattr(self, component_name, None)
            if component is not None and hasattr(component, "to"):
                component = component.to(device)
                setattr(self, component_name, component)
                
        # Note: Diffusion components will be moved as part of the distributed 
        # preparation process, so we don't need to explicitly move them here.

    def _load_condition_models(self):
        """Load text encoders, tokenizers, and image encoder."""
        logger.info("Loading condition models")
        
        components = self.model_specification.load_condition_models()
        logger.info(f"Loaded condition models: {list(components.keys())}")
        
        for name, component in components.items():
            setattr(self, name, component)
            
        logger.info(f"Have image_encoder: {hasattr(self, 'image_encoder')}")
        
        self._are_condition_models_loaded = True

    def _load_latent_models(self):
        """Load VAE for encoding/decoding latents."""
        logger.info("Loading latent models")
        
        components = self.model_specification.load_latent_models()
        
        for name, component in components.items():
            setattr(self, name, component)

    def _load_diffusion_models(self):
        """Load transformer with expanded patch embedding for control channels."""
        logger.info("Loading diffusion models")
        
        # First determine the new in_features size for the transformer
        # Original channels + control channels (doubled for mask)
        input_channels = getattr(self.transformer, "config", self.model_specification.transformer_config).in_channels
        output_channels = input_channels  # Default
        
        if hasattr(self.args, "frame_conditioning_concatenate_mask") and self.args.frame_conditioning_concatenate_mask:
            output_channels = input_channels * 2  # Double channels for mask concatenation
        
        components = self.model_specification.load_diffusion_models(new_in_features=output_channels)
        
        for name, component in components.items():
            setattr(self, name, component)
            
    def _prepare_trainable_parameters(self):
        """Prepare trainable parameters for optimization."""
        logger.info("Preparing trainable parameters")
        parallel_backend = self.state.parallel_backend
        
        # For LoRA training
        if isinstance(self.args, E2VLowRankConfig) or getattr(self.args, "training_type", None) == TrainingType.E2V_LORA:
            self._prepare_lora_parameters()
        # For full fine-tuning
        else:
            self._prepare_full_finetune_parameters()
        
        # Store trainable modules for later use
        self.trainable_modules = [self.transformer]

    def _prepare_lora_parameters(self):
        """Configure model with LoRA parameters."""
        # Configure LoRA
        if not hasattr(self.transformer, "peft_config"):
            # Get LoRA configuration parameters
            rank = self.args.rank
            lora_alpha = self.args.lora_alpha
            target_modules = self.args.target_modules
            
            # Simple validation, consistent with other trainers
            if target_modules is None:
                raise ValueError("target_modules must be specified for LoRA training")
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                init_lora_weights="gaussian",
                lora_dropout=0.0,
                bias="none",
            )
            
            # Convert regex patterns to actual module names
            import re
            
            # Handle either string or list of strings
            patterns = [lora_config.target_modules] if isinstance(lora_config.target_modules, str) else lora_config.target_modules
            
            # Find matching modules
            filtered_modules = []
            for pattern in patterns:
                for name, module in self.transformer.named_modules():
                    if re.search(pattern, name) and hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                        filtered_modules.append(name)
                            
            logger.info(f"Found {len(filtered_modules)} modules matching LoRA target patterns")
            lora_config.target_modules = filtered_modules
            
            # Apply LoRA to the transformer
            from peft import get_peft_model
            
            logger.info(f"Applying LoRA with rank {lora_config.r} and alpha {lora_config.lora_alpha}")
            get_peft_model(self.transformer, lora_config)
            
            # Add QK norm if needed
            if getattr(self.args, "train_qk_norm", False):
                qk_norm_count = 0
                for name, param in self.transformer.named_parameters():
                    if "norm_q" in name or "norm_k" in name:
                        param.requires_grad = True
                        qk_norm_count += 1
                
                logger.info(f"Added {qk_norm_count} QK norm layers to trainable parameters")
        
        # Count trainable parameters (only LoRA params should be trainable)
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        
        # Calculate percentage of trainable parameters
        all_params = [p for p in self.transformer.parameters()]
        total_params = sum(p.numel() for p in all_params)
        trainable_percentage = (total_trainable / total_params) * 100 if total_params > 0 else 0
        
        logger.info(f"Total trainable parameters: {total_trainable:,} ({trainable_percentage:.2f}% of model)")
        self.state.num_trainable_parameters = total_trainable

    def _prepare_full_finetune_parameters(self):
        """Configure model for full fine-tuning."""
        logger.info("Setting up full fine-tuning")
        # Set all transformer parameters to trainable
        for param in self.transformer.parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        
        # Calculate percentage of trainable parameters (for full fine-tuning should be 100%)
        all_params = [p for p in self.transformer.parameters()]
        total_params = sum(p.numel() for p in all_params)
        trainable_percentage = (total_trainable / total_params) * 100 if total_params > 0 else 0
        
        logger.info(f"Total trainable parameters: {total_trainable:,} ({trainable_percentage:.2f}% of model)")
        self.state.num_trainable_parameters = total_trainable
        
    def _prepare_for_training(self):
        """Prepare optimizer, learning rate scheduler and other training components."""
        logger.info("Preparing for training")
        parallel_backend = self.state.parallel_backend
        
        # 1. Gather trainable parameters
        model_parts = self.trainable_modules
        
        # 2. Initialize optimizer using framework approach
        logger.info(f"Initializing {self.args.optimizer} optimizer")
        self.optimizer = optimizer.get_optimizer(
            parallel_backend=self.args.parallel_backend,
            name=self.args.optimizer,
            model_parts=model_parts,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=getattr(self.args, 'beta3', 0.9),
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            fused=False,
        )
        
        # 3. Create learning rate scheduler
        logger.info(f"Creating {self.args.lr_scheduler} learning rate scheduler")
        self.lr_scheduler = optimizer.get_lr_scheduler(
            parallel_backend=self.args.parallel_backend,
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.train_steps,
        )
        
        # 4. Prepare optimizer and scheduler for distributed training
        self.optimizer, self.lr_scheduler = parallel_backend.prepare_optimizer(
            self.optimizer, self.lr_scheduler
        )
        
        # 5. Initialize trackers
        self._init_trackers()

    def _prepare_dataset(self):
        """Initialize dataset and dataloaders."""
        logger.info("Initializing dataset and dataloader")

        # Load dataset config directly from JSON file, matching other trainers
        try:
            with open(self.args.dataset_config, "r") as file:
                dataset_configs = json.load(file)
                if "datasets" not in dataset_configs:
                    raise ValueError(f"'datasets' key not found in config file: {self.args.dataset_config}")
                dataset_configs = dataset_configs["datasets"]
        except Exception as e:
            raise ValueError(f"Failed to load dataset config: {e}")
        
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")

        # Initialize datasets based on config
        datasets = []
        for config in dataset_configs:
            dataset = self._initialize_dataset_from_config(config)
            datasets.append(dataset)

        # Combine datasets with framework's approach
        dataset = data.combine_datasets(
            datasets, 
            buffer_size=self.args.dataset_shuffle_buffer_size, 
            shuffle=True
        )
        
        # Get E2V configuration from the first dataset
        e2v_config = self._extract_e2v_config(self.args.dataset_config)
        
        # Wrap with E2V dataset
        dataset = IterableE2VDataset(
            dataset, 
            e2v_config,
            device=self.state.parallel_backend.device
        )
        
        # Use DPDataLoader to better handle state
        self._prepare_training_dataloader(dataset)
        
        # Handle validation dataset if configured
        if self.args.validation_dataset_file:
            self._prepare_validation_dataset()
        else:
            logger.info("No validation dataset provided")
            self.validation_dataset = None
            self.validation_dataloader = None
    
    def _initialize_dataset_from_config(self, config):
        """Initialize dataset from configuration."""
        # Extract basic dataset parameters
        data_root = config.pop("data_root", None)
        dataset_file = config.pop("dataset_file", None)
        dataset_type = config.pop("dataset_type")
        caption_options = config.pop("caption_options", {})
        
        if data_root is not None and dataset_file is not None:
            raise ValueError("Both data_root and dataset_file cannot be provided in the same dataset config.")
        
        if data_root is None and dataset_file is None:
            raise ValueError("Either data_root or dataset_file must be provided in dataset config.")
        
        # When using video_references, extract reference_suffixes from elements
        if dataset_type == "video_references":
            # Extract suffix patterns from elements configuration
            reference_suffixes = []
            for element in config.get("elements", []):
                if "suffixes" in element:
                    reference_suffixes.extend(element["suffixes"])
                    
            # Add to caption options
            if caption_options is None:
                caption_options = {}
            caption_options["reference_suffixes"] = reference_suffixes
            
        # Initialize dataset using framework pattern
        dataset_name_or_root = data_root or dataset_file
        dataset = data.initialize_dataset(
            dataset_name_or_root, 
            dataset_type, 
            streaming=True, 
            infinite=True, 
            _caption_options=caption_options
        )
        
        # Validate dataset supports precomputation if requested
        if not dataset._precomputable_once and self.args.precomputation_once:
            raise ValueError(
                f"Dataset {dataset_name_or_root} does not support precomputing all embeddings at once."
            )
        
        logger.info(f"Initialized dataset: {dataset_name_or_root}")
        dataset = self.state.parallel_backend.prepare_dataset(dataset)
        dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
        
        return dataset
    
    def _prepare_training_dataloader(self, dataset):
        """Prepare training dataloader from dataset."""
        # Use the same dataloader setup as control_trainer
        parallel_backend = self.state.parallel_backend
        
        # Handle distributed scenario like control_trainer
        local_rank = 0
        if parallel_backend.world_size > 1:
            dp_mesh = parallel_backend.get_mesh("dp_replicate")
            if dp_mesh is not None:
                local_rank = dp_mesh.get_local_rank()
        
        # Use DPDataLoader with hardcoded batch_size=1 like control_trainer
        dataloader = DPDataLoader(
            local_rank,
            dataset,
            batch_size=1,  # Hardcoded to 1 like control_trainer
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items  # Match control_trainer's collate_fn
        )
        
        # Store references
        self.dataset = dataset
        self.dataloader = dataloader
        
    def _prepare_validation_dataset(self):
        """Prepare validation dataset and dataloader."""
        logger.info("Initializing validation dataset")
        
        try:
            # Load validation dataset config
            with open(self.args.validation_dataset_file, "r") as file:
                validation_configs = json.load(file)
                if "datasets" not in validation_configs:
                    raise ValueError(f"'datasets' key not found in config file: {self.args.validation_dataset_file}")
                validation_configs = validation_configs["datasets"]
        except Exception as e:
            raise ValueError(f"Failed to load validation dataset config: {e}")
        
        logger.info(f"Validation configured to use {len(validation_configs)} datasets")
        
        # Initialize validation datasets
        validation_datasets = []
        for config in validation_configs:
            dataset = self._initialize_dataset_from_config(config)
            validation_datasets.append(dataset)
        
        # Combine validation datasets
        validation_dataset = data.combine_datasets(
            validation_datasets, 
            buffer_size=1, 
            shuffle=False
        )
        
        # Get E2V validation configuration
        validation_e2v_config = self._extract_e2v_config(self.args.validation_dataset_file)
        
        # Wrap with E2V validation dataset
        validation_dataset = ValidationE2VDataset(
            validation_dataset,
            validation_e2v_config,
            device=self.state.parallel_backend.device
        )
        
        # Setup validation dataloader exactly like control_trainer
        parallel_backend = self.state.parallel_backend
        
        # Handle distributed scenario (same as control_trainer)
        local_rank = 0
        if parallel_backend.world_size > 1:
            dp_mesh = parallel_backend.get_mesh("dp_replicate")
            if dp_mesh is not None:
                local_rank = dp_mesh.get_local_rank()
        
        # Use DPDataLoader with hardcoded batch_size=1 (exactly like control_trainer)
        validation_dataloader = DPDataLoader(
            local_rank,
            validation_dataset,
            batch_size=1,  # Hardcoded to 1 like control_trainer
            num_workers=self.args.dataloader_num_workers,
            collate_fn=lambda items: items  # Same collate_fn as control_trainer
        )
        
        # Store validation dataset and dataloader
        self.validation_dataset = validation_dataset
        self.validation_dataloader = validation_dataloader

    def _extract_e2v_config(self, config_file):
        """Extract E2V-specific configuration from a dataset config file.
        
        Args:
            config_file: Path to the dataset config JSON file
            
        Returns:
            Dictionary with E2V configuration parameters
        
        Raises:
            ValueError: If required configuration elements are missing
        """
        # Initialize E2V config with empty dict (no defaults)
        e2v_config = {}
        
        # Read the first config from the file
        try:
            with open(config_file, "r") as file:
                config_data = json.load(file)
                if "datasets" not in config_data or not config_data["datasets"]:
                    raise ValueError(f"No datasets found in configuration file: {config_file}")
                
                first_config = config_data["datasets"][0]
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_file}: {e}")
        
        # Copy essential configuration parts
        for key in ["elements", "processors", "data_root", "tensor_combinations", "visualization"]:
            if key in first_config:
                e2v_config[key] = first_config[key]
        
        # Validate the configuration using shared function
        validate_e2v_config(e2v_config)
        
        # Log the tensor combinations configuration
        if "tensor_combinations" in e2v_config:
            logger.info(f"Using tensor combinations configuration: {e2v_config['tensor_combinations']}")
        
        # Derive reference_suffixes from elements if not explicitly provided
        if "reference_suffixes" in first_config:
            e2v_config["reference_suffixes"] = first_config["reference_suffixes"]
        elif "elements" in e2v_config:
            # Extract all suffixes from the elements
            all_suffixes = []
            for element in e2v_config["elements"]:
                if "suffixes" in element:
                    all_suffixes.extend(element["suffixes"])
            e2v_config["reference_suffixes"] = all_suffixes
        
        # Get frame conditioning parameters from config or args
        if "frame_conditioning_type" in first_config:
            e2v_config["frame_conditioning_type"] = first_config["frame_conditioning_type"]
        else:
            e2v_config["frame_conditioning_type"] = getattr(self.args, "frame_conditioning_type", "full")
            
        if "frame_conditioning_index" in first_config:
            e2v_config["frame_conditioning_index"] = first_config["frame_conditioning_index"]
        else:
            e2v_config["frame_conditioning_index"] = getattr(self.args, "frame_conditioning_index", 0)
            
        if "frame_conditioning_concatenate_mask" in first_config:
            e2v_config["frame_conditioning_concatenate_mask"] = first_config["frame_conditioning_concatenate_mask"]
        else:
            e2v_config["frame_conditioning_concatenate_mask"] = getattr(self.args, "frame_conditioning_concatenate_mask", True)
        
        return e2v_config
        
    def _prepare_checkpointing(self):
        """Set up checkpointing for the trainer."""
        parallel_backend = self.state.parallel_backend
        
        def save_model_hook(state_dict):
            """Hook called during checkpoint saving."""
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if self.args.training_type == TrainingType.E2V_LORA:
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    # Prepare metadata for LoRA
                    metadata = {
                        "r": self.args.rank,
                        "lora_alpha": self.args.lora_alpha,
                        "init_lora_weights": True,
                        "target_modules": self.args.target_modules,
                    }
                    metadata = {"lora_config": json.dumps(metadata, indent=4)}
                    
                    self.model_specification._save_lora_weights(
                        self.args.output_dir, state_dict, self.scheduler, metadata
                    )
                elif self.args.training_type == TrainingType.E2V_FULL_FINETUNE:
                    self.model_specification._save_model(
                        self.args.output_dir, self.transformer, state_dict, self.scheduler
                    )
            parallel_backend.wait_for_everyone()
            
        # Use the parallel backend's checkpointer
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
        
        # Handle resuming from checkpoint
        resume_from_checkpoint = self.args.resume_from_checkpoint
        if resume_from_checkpoint == "latest":
            resume_from_checkpoint = -1
        if resume_from_checkpoint is not None:
            self.checkpointer.load(resume_from_checkpoint)
            
    def _train(self):
        """Run the training loop."""
        logger.info("Starting training")
        
        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        
        # For IterableDatasets, we can't reliably get the length,
        # but we can use the desired number of training steps directly
        total_batch_size = self.args.train_batch_size * parallel_backend.world_size * self.state.gradient_accumulation_steps
        logger.info(f"  Batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.state.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.train_steps}")
        
        # Initialize training state and counters
        current_step = train_state.step
        max_steps = self.args.train_steps
        train_state.max_steps = max_steps  # Store for reference
        
        # Initialize counters like control_trainer
        if not hasattr(train_state, "observed_data_samples"):
            train_state.observed_data_samples = 0
        if not hasattr(train_state, "observed_num_tokens"):
            train_state.observed_num_tokens = 0
        if not hasattr(train_state, "log_steps"):
            train_state.log_steps = []
        
        # Create progress bar
        progress_bar = tqdm(
            range(current_step, max_steps),
            disable=not self.state.parallel_backend.is_local_main_process,
            desc="Training steps",
        )
        
        # Initialize data iterator
        data_iterator = iter(self.dataloader)
        
        # Run training loop until we reach max steps or run out of data
        while current_step < max_steps:
            try:
                # Get next batch
                batch = next(data_iterator)
            except StopIteration:
                # Restart iterator if we run out of data
                logger.info("Reached end of dataset, restarting data iterator")
                data_iterator = iter(self.dataloader)
                batch = next(data_iterator)
                
            # Handle batch formatting like control_trainer
            # DPDataLoader with collate_fn=lambda items: items returns a list
            batch = batch[0] if isinstance(batch, list) else batch
            
            # Move batch to correct device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            loss = self._forward_pass(batch)
            
            # For gradient accumulation, scale loss to match total batch size
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
                
            # Backward pass
            loss.backward()
            
            # Check for NaN/Inf
            if self.state.logging_nan_or_inf:
                self._check_for_nan_in_loss_and_grads(self.trainable_modules)
            
            # Parameter update
            # Control trainer checks step % gradient_accumulation_steps == 0
            if current_step % self.args.gradient_accumulation_steps == 0:
                self._update_parameters()
                    
                # Update counters and trackers
                progress_bar.update(1)
                current_step += 1
                train_state.step = current_step  # Update the step in train_state
                train_state.observed_data_samples += self.args.batch_size * parallel_backend._dp_degree
                
                # Track token count
                try:
                    # Get latents from batch to calculate tokens
                    latents = batch.get("latents")
                    if latents is not None:
                        # Get patch size from transformer config if available
                        patch_size = self._get_patch_size()
                        
                        # Calculate tokens
                        train_state.observed_num_tokens += math.prod(latents.shape[:-1]) // patch_size
                except Exception as e:
                    # Don't break training if token tracking fails
                    logger.warning(f"Failed to track tokens: {e}")
                
                # Log metrics
                if parallel_backend.is_main_process:
                    # Prepare metrics
                    metrics = {
                        "loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else 0,
                        "step": current_step,
                        "observed_data_samples": train_state.observed_data_samples,
                        "observed_num_tokens": train_state.observed_num_tokens
                    }
                    
                    # Log step information
                    logger.info(f"Step {current_step}: loss = {metrics['loss']:.4f}, lr = {metrics['lr']:.6f}")
                    
                    # Log to trackers at regular intervals
                    if current_step % self.args.logging_steps == 0:
                        parallel_backend.log(metrics, step=current_step)
                        train_state.log_steps.append(current_step)
                
                # Run validation if configured
                if self.validation_dataloader is not None and \
                   self.args.validation_steps > 0 and \
                   current_step % self.args.validation_steps == 0:
                    self._validate(step=current_step, final_validation=False)
                
                # Create checkpoint if configured
                if self.checkpointer and self.checkpointer.should_save(current_step):
                    self.checkpointer.save()
        
        # Make sure we create a final checkpoint
        if current_step > 0 and self.checkpointer:
            self.checkpointer.save()
            
    def _get_patch_size(self):
        """Get patch size from transformer configuration."""
        patch_size = 1
        if hasattr(self.transformer, "config"):
            if hasattr(self.transformer.config, "patch_size") and hasattr(self.transformer.config, "patch_size_t"):
                patch_size = self.transformer.config.patch_size * self.transformer.config.patch_size_t
            elif isinstance(getattr(self.transformer.config, "patch_size", None), int):
                patch_size = self.transformer.config.patch_size
            elif isinstance(getattr(self.transformer.config, "patch_size", None), (list, tuple)):
                patch_size = math.prod(self.transformer.config.patch_size)
        return patch_size
        
    def _validate(self, step: int = None, final_validation: bool = False):
        """Run validation.
        
        Args:
            step: Current training step (if None, uses train_state.step)
            final_validation: Whether this is the final validation run
        """
        if self.validation_dataloader is None:
            return
        
        logger.info("Running validation")
        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        
        # Use provided step or get from train_state
        if step is None:
            step = train_state.step
        
        with torch.no_grad():
            total_val_loss = 0.0
            val_steps = 0
            
            for i, batch in enumerate(self.validation_dataloader):
                # Only process a few batches
                if i >= self.args.max_validation_batches:
                    break
                
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss = self._forward_pass(batch)
                total_val_loss += loss.detach().item()
                val_steps += 1
            
            # Calculate average validation loss
            if val_steps > 0:
                avg_val_loss = total_val_loss / val_steps
                
                # Log validation metrics
                if parallel_backend.is_main_process:
                    logger.info(f"Validation loss: {avg_val_loss:.4f}")
                    
                    metrics = {
                        "val_loss": avg_val_loss,
                        "step": step,
                    }
                    
                    # Log to trackers
                    parallel_backend.log(metrics, step=step)
                
                # Generate sample if requested and we have validation data
                if self.args.validation_generate_samples and self.validation_dataloader is not None:
                    try:
                        batch = next(iter(self.validation_dataloader))
                        batch = self._move_batch_to_device(batch)
                        self._generate_samples(batch)
                    except (StopIteration, Exception) as e:
                        logger.warning(f"Failed to generate validation sample: {e}")
                        
    def _generate_samples(self, batch):
        """Generate video samples using the current model."""
        # This would be nice to have but is beyond the scope of this initial implementation
        # Would need to create a proper pipeline that supports E2V generation
        pass

    def _encode_elements(self, preprocessed_elements):
        """Process preprocessed elements through models.
        
        This method:
        1. Uses registered encoder functions for different processor types
        2. Processes each element through the appropriate encoder
        3. Returns all encoded features for combining
        
        Args:
            preprocessed_elements: Dictionary of preprocessed tensors by processor type
            
        Returns:
            Dictionary of encoded features by processor type
        """
        encoded_features = {}
        
        logger.debug(f"Processing {len(preprocessed_elements)} processor types: {list(preprocessed_elements.keys())}")
        
        # Process elements for each registered processor type
        for proc_name, elements in preprocessed_elements.items():
            # Skip if no encoder registered for this processor type
            if proc_name not in ENCODER_REGISTRY:
                logger.warning(f"No encoder registered for processor type: {proc_name}")
                continue
            
            logger.debug(f"Processing {proc_name} with {len(elements)} elements: {list(elements.keys())}")
            
            # Get appropriate model for this processor type
            model = None
            if proc_name == "vae":
                model = self.vae
            elif proc_name == "clip":
                model = self.image_encoder
            
            if model is None:
                logger.warning(f"No model available for processor type: {proc_name}")
                continue
            
            # Initialize processor entry in encoded_features
            encoded_features[proc_name] = {}
            
            # Get encoder function
            encoder_func = ENCODER_REGISTRY[proc_name]
            
            # Process each element
            for element_name, element_info in elements.items():
                logger.debug(f"Encoding {element_name} with {proc_name}")
                
                if "tensor" not in element_info:
                    logger.error(f"No tensor found for {element_name} in {proc_name} processor")
                    continue
                    
                if "config" not in element_info:
                    logger.warning(f"No config found for {element_name} in {proc_name} processor, using empty config")
                    element_info["config"] = {}
                    
                # Log tensor details
                tensor = element_info["tensor"]
                logger.debug(f"Element tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
                
                # Encode element
                try:
                    result = encoder_func(element_info["tensor"], model, element_info["config"])
                    encoded_features[proc_name][element_name] = result
                    logger.debug(f"Successfully encoded {element_name} with {proc_name}")
                except Exception as e:
                    logger.error(f"Error encoding {element_name} with {proc_name}: {e}", exc_info=True)
                    # Continue with other elements
                    continue
        
        logger.debug(f"Encoded features for processors: {list(encoded_features.keys())}")
        return encoded_features

    def _combine_features(self, encoded_features, tensor_combinations):
        """Combine encoded features according to tensor_combinations configuration.
        
        This method:
        1. For each output tensor defined in tensor_combinations
        2. For each processor contributing to that tensor
        3. Use the appropriate combiner function
        4. Combine results from different processors if needed
        
        Args:
            encoded_features: Dictionary of encoded features by processor type
            tensor_combinations: Configuration for combining tensors
            
        Returns:
            Dictionary of combined tensors for model input
        """
        combined_tensors = {}
        
        logger.debug(f"Combining features for {len(tensor_combinations)} outputs: {list(tensor_combinations.keys())}")
        logger.debug(f"Available encoded features: {list(encoded_features.keys())}")
        
        # For each output tensor defined in tensor_combinations
        for output_name, processor_list in tensor_combinations.items():
            processor_results = {}
            logger.debug(f"Combining {len(processor_list)} processors for output '{output_name}': {processor_list}")
            
            # For each processor contributing to this output
            for proc_name in processor_list:
                # Skip if processor not in encoded features
                if proc_name not in encoded_features:
                    logger.warning(f"Processor '{proc_name}' specified in tensor_combinations but not in encoded features")
                    continue
                    
                # Skip if no combiner registered for this processor
                if proc_name not in COMBINER_REGISTRY:
                    logger.warning(f"No combiner registered for processor type: '{proc_name}'")
                    continue
                
                # Get features for this processor
                processor_features = encoded_features[proc_name]
                if not processor_features:
                    logger.warning(f"No features for processor '{proc_name}'")
                    continue
                
                logger.debug(f"Processor '{proc_name}' has features for elements: {list(processor_features.keys())}")
                
                # Get combiner function
                combiner_func = COMBINER_REGISTRY[proc_name]
                
                # Combine features using the registered combiner
                try:
                    logger.debug(f"Combining features for '{proc_name}'")
                    result = combiner_func(processor_features)
                    
                    if result is not None:
                        processor_results[proc_name] = result
                        if isinstance(result, torch.Tensor):
                            logger.debug(f"Combined '{proc_name}' result shape: {result.shape}")
                        else:
                            logger.debug(f"Combined '{proc_name}' result type: {type(result)}")
                    else:
                        logger.warning(f"Combiner for '{proc_name}' returned None")
                except Exception as e:
                    logger.error(f"Error combining features for '{proc_name}': {e}", exc_info=True)
                    # Continue with other processors
                    continue
            
            # Create output tensor based on number of processors
            if len(processor_results) == 1:
                # Only one processor, use its result directly
                proc_name = list(processor_results.keys())[0]
                combined_tensors[output_name] = processor_results[proc_name]
                logger.debug(f"Using direct result from '{proc_name}' for output '{output_name}'")
            elif len(processor_results) > 1:
                # Multiple processors, concatenate along channel dimension
                try:
                    tensors = list(processor_results.values())
                    logger.debug(f"Concatenating {len(tensors)} tensors for '{output_name}'")
                    logger.debug(f"Tensor shapes: {[t.shape for t in tensors if isinstance(t, torch.Tensor)]}")
                    
                    combined_tensors[output_name] = torch.cat(tensors, dim=1)  # Channel dimension
                    logger.debug(f"Combined tensor shape for '{output_name}': {combined_tensors[output_name].shape}")
                except Exception as e:
                    logger.error(f"Error concatenating results for '{output_name}': {e}", exc_info=True)
                    logger.error(f"Tensor shapes: {[t.shape for t in list(processor_results.values()) if isinstance(t, torch.Tensor)]}")
                    # Skip this output
                    continue
            else:
                # No results, log warning
                logger.warning(f"No valid results for output '{output_name}'")
        
        logger.debug(f"Combined tensors for outputs: {list(combined_tensors.keys())}")
        return combined_tensors

    def _forward_pass(self, batch):
        """Run forward pass with E2V conditioning.
        
        This method:
        1. Processes preprocessed tensors through models (VAE, CLIP)
        2. Combines encoded features according to tensor_combinations 
        3. Runs the model forward pass with prepared conditions
        
        This approach avoids CUDA multiprocessing issues by keeping all
        model inference in the main process.
        
        Args:
            batch: Training batch with preprocessed elements
            
        Returns:
            Loss tensor
        """
        # Log batch structure
        logger.debug(f"Batch keys: {list(batch.keys())}")
        
        # Process inputs
        text_embeddings = batch.get("text_embeddings")
        video_latents = batch.get("latents")
        latents_mean = batch.get("latents_mean", None)
        latents_std = batch.get("latents_std", None)
        
        if text_embeddings is None:
            logger.error("No text_embeddings found in batch")
        else:
            logger.debug(f"Text embeddings shape: {text_embeddings.shape}")
            
        if video_latents is None:
            logger.error("No video latents found in batch")
        else:
            logger.debug(f"Video latents shape: {video_latents.shape}")
        
        # Get preprocessed elements and configs
        preprocessed_elements = batch.get("preprocessed_elements", {})
        if not preprocessed_elements:
            logger.warning("No preprocessed_elements found in batch")
        else:
            logger.debug(f"Preprocessed element types: {list(preprocessed_elements.keys())}")
            for proc_type, elements in preprocessed_elements.items():
                logger.debug(f"  {proc_type} has {len(elements)} elements: {list(elements.keys())}")
        
        # Get tensor_combinations from batch (provided by dataset)
        tensor_combinations = batch.get("tensor_combinations")
        if tensor_combinations is None:
            logger.error("No tensor_combinations found in batch data")
            raise ValueError(
                "No tensor_combinations found in batch data. "
                "E2V training requires explicit tensor_combinations configuration."
            )
        
        # Validate tensor combinations before encoding
        validate_tensor_combinations(tensor_combinations)
        logger.debug(f"Using tensor_combinations: {tensor_combinations}")
        
        # Process preprocessed elements through models
        logger.debug("Beginning element encoding")
        encoded_features = self._encode_elements(preprocessed_elements)
        
        # Combine features according to tensor_combinations
        logger.debug("Combining encoded features")
        combined_tensors = self._combine_features(encoded_features, tensor_combinations)
        
        # Log combined tensor information
        logger.debug(f"Combined tensor keys: {list(combined_tensors.keys())}")
        
        # Find control_latents (required for model)
        control_latents = find_tensor_by_key_pattern(combined_tensors, "condition_latents")
        if control_latents is None:
            logger.error("No condition_latents found in combined tensors")
            # Check what keys are available to help debug
            available_keys = list(combined_tensors.keys())
            logger.error(f"Available keys in combined_tensors: {available_keys}")
            
            # Check if we can provide detailed information on what was attempted
            logger.error(f"Tensor combinations configuration: {tensor_combinations}")
            logger.error(f"Encoded feature types: {list(encoded_features.keys())}")
            
            raise ValueError(
                "No 'condition_latents' found in combined tensors. "
                "Please check your tensor_combinations configuration."
            )
        else:
            logger.debug(f"Control latents shape: {control_latents.shape}")
        
        # Find clip_embeddings (optional)
        clip_embeddings = find_tensor_by_key_pattern(combined_tensors, "encoder_hidden_states") or \
                        find_tensor_by_key_pattern(combined_tensors, "clip")
        if clip_embeddings is not None:
            logger.debug(f"CLIP embeddings shape: {clip_embeddings.shape}")
        else:
            logger.debug("No CLIP embeddings found")
        
        # Generate random sigmas for flow matching
        generator = torch.Generator(device=self.state.parallel_backend.device).manual_seed(self.args.seed)
        batch_size = video_latents.shape[0]
        
        # Prepare batch for model
        latent_model_conditions = {
            "latents": video_latents,
            "control_latents": control_latents,
        }
        
        if latents_mean is not None and latents_std is not None:
            latent_model_conditions["latents_mean"] = latents_mean
            latent_model_conditions["latents_std"] = latents_std
        
        # Condition model
        condition_model_conditions = {
            "encoder_hidden_states": text_embeddings,
        }
        
        # Add CLIP embeddings if available
        if clip_embeddings is not None:
            condition_model_conditions["encoder_hidden_states_image"] = clip_embeddings
        
        # Sample sigmas for training
        sigmas = torch.randn(
            (batch_size,),
            device=self.state.parallel_backend.device,
            generator=generator,
        ).abs_()
        sigmas = sigmas.view(-1, 1, 1, 1, 1)
        
        logger.debug("Running model forward pass")
        # Forward through model specification
        try:
            loss = self.model_specification.forward(
                transformer=self.transformer,
                condition_model_conditions=condition_model_conditions,
                latent_model_conditions=latent_model_conditions,
                sigmas=sigmas,
                generator=generator,
            )
            logger.debug(f"Forward pass complete, loss: {loss.item()}")
            return loss
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}", exc_info=True)
            logger.error(f"condition_model_conditions keys: {list(condition_model_conditions.keys())}")
            logger.error(f"latent_model_conditions keys: {list(latent_model_conditions.keys())}")
            # Re-raise to stop training
            raise
        
    def _update_parameters(self):
        """Update model parameters with the optimizer."""
        # Clip gradients
        if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
            # Use utils helper for gradient clipping
            model_parts = [self.transformer]
            grad_norm = utils.torch._clip_grad_norm_while_handling_failing_dtensor_cases(
                [p for m in model_parts for p in m.parameters()],
                self.args.clip_grad_norm,
                foreach=True,
                pp_mesh=self.state.parallel_backend.get_mesh("pp") 
                    if self.state.parallel_backend.pipeline_parallel_enabled 
                    else None,
            )
        
        # Take optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
    def _move_batch_to_device(self, batch):
        """Move batch to the correct device."""
        device = self.state.parallel_backend.device
        
        def _move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: _move(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_move(o) for o in obj]
            else:
                return obj
        
        return _move(batch)
    
    def _check_for_nan_in_loss_and_grads(self, modules):
        """Check for NaN/Inf in loss and gradients."""
        for module in modules:
            for name, param in module.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any().item():
                        logger.warning(f"NaN detected in gradient for {name}")
                    if torch.isinf(param.grad).any().item():
                        logger.warning(f"Inf detected in gradient for {name}")