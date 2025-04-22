"""E2V (Elements-to-Video) Trainer implementation.

This module implements the E2V trainer, which handles training Wan models
using multiple reference images (elements) as conditioning for video generation.

The trainer follows the separation of concerns pattern where:
1. Preprocessing is done in the dataset (IterableE2VDataset)
2. Model inference is done in the trainer (this file)
3. Configuration is strictly driven by the config file

No legacy code support or implicit defaults are provided - all functionality
is driven explicitly by configuration values.
"""
import functools
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import datasets.distributed
import safetensors.torch
import torch
import torch.backends
import wandb
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_layerwise_casting
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict
from tqdm import tqdm

from finetrainers import data, logging, optimizer, parallel, patches, utils
from finetrainers.config import TrainingType
from finetrainers.patches import load_lora_weights
from finetrainers.state import State, TrainState

from .config import E2VFullRankConfig, E2VLowRankConfig
from .data import IterableE2VDataset, ValidationE2VDataset
from .encoders import ENCODER_REGISTRY, encode_vae, encode_clip
from .combiners import COMBINER_REGISTRY, combine_vae_features, combine_clip_features, get_encoder, get_combiner
from .utils import validate_e2v_config, validate_tensor_combinations, find_tensor_by_key_pattern


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs
    from finetrainers.models import ControlModelSpecification

ArgsType = Union["BaseArgs", E2VFullRankConfig, E2VLowRankConfig]

logger = get_logger()


class E2VTrainer:
    # fmt: off
    _all_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "unet", "vae", "scheduler", "image_encoder"]
    _condition_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]
    _latent_component_names = ["vae"]
    _diffusion_component_names = ["transformer", "unet", "scheduler"]
    # fmt: on

    def __init__(self, args: ArgsType, model_specification: "ControlModelSpecification") -> None:
        """Initialize the E2V trainer.
        
        Args:
            args: Training arguments
            model_specification: Model specification
        """
        self.args = args
        self.state = State()
        self.state.train_state = TrainState()

        # Initialize components to None
        # Tokenizers
        self.tokenizer = None
        self.tokenizer_2 = None
        self.tokenizer_3 = None

        # Text encoders
        self.text_encoder = None
        self.text_encoder_2 = None
        self.text_encoder_3 = None
        
        # Image encoder for CLIP pathway
        self.image_encoder = None

        # Denoisers
        self.transformer = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        # Optimizer & LR scheduler
        self.optimizer = None
        self.lr_scheduler = None

        # Checkpoint manager
        self.checkpointer = None
        
        # Training state
        self.state.num_trainable_parameters = 0
        
        # Initialize distributed training first
        self._init_distributed()
        self._init_config_options()
        
        # Initialize logging and directories
        self._init_logging()
        self._init_directories_and_repositories()

        # Perform any patches that might be necessary for training to work as expected
        patches.perform_patches_for_training(self.args, self.state.parallel_backend)

        self.model_specification = model_specification
        self._are_condition_models_loaded = False

        # Pass frame conditioning parameters to model specification
        # Use getattr with default values for potentially missing parameters
        frame_conditioning_type = getattr(args, "frame_conditioning_type", "full")
        frame_conditioning_index = getattr(args, "frame_conditioning_index", 0)
        frame_conditioning_concatenate_mask = getattr(args, "frame_conditioning_concatenate_mask", True)
        
        model_specification._trainer_init(
            frame_conditioning_type, frame_conditioning_index, frame_conditioning_concatenate_mask
        )
        
        # We'll initialize trackers in _prepare_for_training, not here

    def run(self) -> None:
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
                self._validate()
                
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

    def _init_distributed(self) -> None:
        """Initialize distributed training."""
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

        backend_cls: parallel.ParallelBackendType = parallel.get_parallel_backend_cls(self.args.parallel_backend)
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

    def _init_config_options(self) -> None:
        """Initialize configuration options."""
        # Set up configuration options
        self.state.gradient_accumulation_steps = self.args.gradient_accumulation_steps

        # Use getattr with default False in case the attribute doesn't exist in BaseArgs
        self.state.logging_nan_or_inf = getattr(self.args, "logging_nan_or_inf", False)
        self.state.allow_tf32 = self.args.allow_tf32
        if self.state.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if not hasattr(self.args, "train_batch_size") or not self.args.train_batch_size:
            self.args.train_batch_size = 1

        if not hasattr(self.args, "eval_batch_size") or not self.args.eval_batch_size:
            self.args.eval_batch_size = 1

        if hasattr(self.args, "setup_torch_compile") and self.args.setup_torch_compile:
            # Reset compilation cache
            os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
            
    def _init_logging(self) -> None:
        """Initialize logging functionality."""
        # Only log once from the main process
        if self.state.parallel_backend.is_main_process:
            logger.info(f"E2V training: {self.args.training_type}")
            logger.info(f"Output directory: {self.args.output_dir}")
            
            # Log some important args
            logger.info(f"Training batch size: {self.args.train_batch_size}")
            logger.info(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
            
            # E2V is configuration-driven
            if self.args.training_type == TrainingType.E2V_LORA:
                # Access values directly from args
                logger.info(f"LoRA rank: {self.args.rank}")
                logger.info(f"LoRA alpha: {self.args.lora_alpha}")
                logger.info(f"LoRA target modules: {self.args.target_modules}")
    
    def _init_trackers(self) -> None:
        """Initialize model trackers like WandB."""
        parallel_backend = self.state.parallel_backend
        
        # Follow the same pattern as other trainers for consistency
        trackers = [self.args.report_to]
        experiment_name = getattr(self.args, "tracker_name", None) or "finetrainers-experiment"
        parallel_backend.initialize_trackers(
            trackers, 
            experiment_name=experiment_name, 
            config=self._get_training_info(), 
            log_dir=self.args.logging_dir
        )
        
    def _get_training_info(self) -> Dict[str, Any]:
        """Get training information for logging."""
        info = self.args.to_dict()

        # Removing flow matching arguments when not using flow-matching objective
        diffusion_args = info.get("diffusion_arguments", {})
        scheduler_name = self.scheduler.__class__.__name__ if self.scheduler is not None else ""
        if scheduler_name != "FlowMatchEulerDiscreteScheduler":
            filtered_diffusion_args = {k: v for k, v in diffusion_args.items() if "flow" not in k}
        else:
            filtered_diffusion_args = diffusion_args

        info.update({"diffusion_arguments": filtered_diffusion_args})
        return info
    
    def _init_directories_and_repositories(self) -> None:
        """Initialize output directories and repositories."""
        if self.state.parallel_backend.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = Path(self.args.output_dir)

            if self.args.push_to_hub:
                repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
                self.state.repo_id = create_repo(token=self.args.hub_token, repo_id=repo_id, exist_ok=True).repo_id

    def _prepare_models(self) -> None:
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
            
    def _move_components_to_device(self) -> None:
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

    def _load_condition_models(self) -> None:
        """Load text encoders, tokenizers, and image encoder."""
        logger.info("Loading condition models")
        
        components = self.model_specification.load_condition_models()
        logger.info(f"Loaded condition models: {list(components.keys())}")
        
        for name, component in components.items():
            setattr(self, name, component)
            
        logger.info(f"Have image_encoder: {hasattr(self, 'image_encoder')}")
        if hasattr(self, 'image_encoder'):
            logger.info(f"image_encoder type: {type(self.image_encoder)}")
        else:
            logger.warning("No image_encoder loaded - E2V training requires CLIP image_encoder")
            logger.warning("This might be because we're not using WanE2VModelSpecification")
            logger.warning(f"Using model specification type: {type(self.model_specification)}")
        
        self._are_condition_models_loaded = True

    def _load_latent_models(self) -> None:
        """Load VAE for encoding/decoding latents."""
        logger.info("Loading latent models")
        
        components = self.model_specification.load_latent_models()
        
        for name, component in components.items():
            setattr(self, name, component)

    def _load_diffusion_models(self) -> None:
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

    def _prepare_trainable_parameters(self) -> None:
        """Prepare trainable parameters for optimization."""
        logger.info("Preparing trainable parameters")
        parallel_backend = self.state.parallel_backend
        
        # For LoRA training
        if isinstance(self.args, E2VLowRankConfig) or getattr(self.args, "training_type", None) == TrainingType.E2V_LORA:
            # Configure LoRA
            if not hasattr(self.transformer, "peft_config"):
                # Access attributes directly from self.args
                # These values should be available after being properly mapped in the config.py
                rank = self.args.rank
                lora_alpha = self.args.lora_alpha
                target_modules = self.args.target_modules
                
                # Simple validation, consistent with other trainers
                if target_modules is None:
                    raise ValueError("target_modules must be specified for LoRA training")
                
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
                
                from peft import get_peft_model
                
                # Apply LoRA to the transformer
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
            
            # Set training modules and mark lora parameters as trainable
            logger.info("Setting up LoRA training parameters")
            
            # Count trainable parameters (only LoRA params should be trainable)
            trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
            total_trainable = sum(p.numel() for p in trainable_params)
            
            # Calculate percentage of trainable parameters
            all_params = [p for p in self.transformer.parameters()]
            total_params = sum(p.numel() for p in all_params)
            trainable_percentage = (total_trainable / total_params) * 100 if total_params > 0 else 0
            
            logger.info(f"Total trainable parameters: {total_trainable:,} ({trainable_percentage:.2f}% of model)")
            self.state.num_trainable_parameters = total_trainable
        
        # For full fine-tuning
        else:
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
        
        # Store trainable modules for later use
        self.trainable_modules = [self.transformer]

    def _prepare_for_training(self) -> None:
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

    def _prepare_dataset(self) -> None:
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
                dataset_name_or_root, dataset_type, streaming=True, infinite=True, _caption_options=caption_options
            )
            
            # Validate dataset supports precomputation if requested
            if not dataset._precomputable_once and self.args.precomputation_once:
                raise ValueError(
                    f"Dataset {dataset_name_or_root} does not support precomputing all embeddings at once."
                )
            
            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
            datasets.append(dataset)

        # Combine datasets with framework's approach
        dataset = data.combine_datasets(datasets, buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True)
        
        # Get E2V configuration from the first dataset
        e2v_config = self._extract_e2v_config(self.args.dataset_config)
        
        # Wrap with E2V dataset
        dataset = IterableE2VDataset(
            dataset, 
            e2v_config,
            device=self.state.parallel_backend.device
        )
        
        # Create dataloader using framework pattern
        dataloader = self.state.parallel_backend.prepare_dataloader(
            dataset, 
            batch_size=self.args.train_batch_size, 
            num_workers=self.args.dataloader_num_workers, 
            pin_memory=self.args.pin_memory
        )
        
        # Use the same variable names as other trainers
        self.dataset = dataset
        self.dataloader = dataloader
        
        # Handle validation dataset if configured
        if self.args.validation_dataset_file:
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
                # Extract basic dataset parameters
                data_root = config.pop("data_root", None)
                dataset_file = config.pop("dataset_file", None)
                dataset_type = config.pop("dataset_type")
                caption_options = config.pop("caption_options", {})
                
                if data_root is not None and dataset_file is not None:
                    raise ValueError("Both data_root and dataset_file cannot be provided in the same validation config.")
                
                if data_root is None and dataset_file is None:
                    raise ValueError("Either data_root or dataset_file must be provided in validation config.")
                
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
                
                # Initialize validation dataset
                dataset_name_or_root = data_root or dataset_file
                validation_dataset = data.initialize_dataset(
                    dataset_name_or_root, dataset_type, streaming=True, infinite=False, _caption_options=caption_options
                )
                
                logger.info(f"Initialized validation dataset: {dataset_name_or_root}")
                validation_dataset = self.state.parallel_backend.prepare_dataset(validation_dataset)
                validation_dataset = data.wrap_iterable_dataset_for_preprocessing(validation_dataset, dataset_type, config)
                validation_datasets.append(validation_dataset)
            
            # Combine validation datasets
            validation_dataset = data.combine_datasets(validation_datasets, buffer_size=1, shuffle=False)
            
            # Get E2V validation configuration
            validation_e2v_config = self._extract_e2v_config(self.args.validation_dataset_file)
            
            # Wrap with E2V validation dataset
            validation_dataset = ValidationE2VDataset(
                validation_dataset,
                validation_e2v_config,
                device=self.state.parallel_backend.device
            )
            
            # Create validation dataloader
            validation_dataloader = self.state.parallel_backend.prepare_dataloader(
                validation_dataset,
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.pin_memory
            )
            
            # Store validation dataset and dataloader
            self.validation_dataset = validation_dataset
            self.validation_dataloader = validation_dataloader
        else:
            logger.info("No validation dataset provided")
            self.validation_dataset = None
            self.validation_dataloader = None

    def _extract_e2v_config(self, config_file: str) -> Dict[str, Any]:
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

    def _prepare_checkpointing(self) -> None:
        """Set up checkpointing for the trainer."""
        parallel_backend = self.state.parallel_backend
        
        def save_model_hook(state_dict: Dict[str, Any]) -> None:
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if self.args.training_type == TrainingType.E2V_LORA:
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    # fmt: off
                    metadata = {
                        "r": self.args.rank,
                        "lora_alpha": self.args.lora_alpha,
                        "init_lora_weights": True,
                        "target_modules": self.args.target_modules,
                    }
                    metadata = {"lora_config": json.dumps(metadata, indent=4)}
                    # fmt: on
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

    def _train(self) -> None:
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
        
        # Set initial values
        train_state.global_step = train_state.global_step or 0
        train_state.max_steps = self.args.train_steps
        
        progress_bar = tqdm(
            range(train_state.global_step, self.args.train_steps),
            disable=not self.state.parallel_backend.is_local_main_process,
            desc="Training steps",
        )
        
        # Use a simpler approach that directly loops until we reach the target number of steps
        data_iterator = iter(self.dataloader)
        
        while train_state.global_step < self.args.train_steps:
            try:
                # Get next batch
                batch = next(data_iterator)
            except StopIteration:
                # Restart iterator if we run out of data
                logger.info("Reached end of dataset, restarting data iterator")
                data_iterator = iter(self.dataloader)
                batch = next(data_iterator)
                
            with self.state.parallel_backend.accumulate():
                # Move batch to correct device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss = self._forward_pass(batch)
                
                # Backward pass
                self.state.parallel_backend.backward(loss)
                
                # Check for NaN/Inf
                if self.state.logging_nan_or_inf:
                    self._check_for_nan_in_loss_and_grads(self.trainable_modules)
                
                # Parameter update
                if self.state.parallel_backend.sync_gradients:
                    self._update_parameters()
                    
                    progress_bar.update(1)
                    train_state.global_step += 1
                    
                    # Log metrics
                    if parallel_backend.is_main_process:
                        metrics = {
                            "loss": loss.detach().item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": train_state.global_step,
                        }
                        logger.info(f"Step {train_state.global_step}: loss = {metrics['loss']:.4f}, lr = {metrics['lr']:.6f}")
                        
                        # Log to trackers
                        parallel_backend.log_metrics(metrics, train_state.global_step)
                    
                    # Run validation if configured
                    if self.validation_dataloader is not None and \
                       self.args.validation_steps > 0 and \
                       train_state.global_step % self.args.validation_steps == 0:
                        self._validate()
                    
                    # Create checkpoint if configured
                    if self.checkpointer and self.checkpointer.should_save(train_state.global_step):
                        self.checkpointer.save()
        
        # Make sure we create a final checkpoint
        if train_state.global_step > 0 and self.checkpointer:
            self.checkpointer.save()
            
            # Upload to Hugging Face Hub if specified
            if self.args.push_to_hub and self.state.parallel_backend.is_main_process:
                self._upload_to_hub()

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
        
        # Process elements for each registered processor type
        for proc_name, elements in preprocessed_elements.items():
            # Skip if no encoder registered for this processor type
            if proc_name not in ENCODER_REGISTRY:
                logger.warning(f"No encoder registered for processor type: {proc_name}")
                continue
            
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
                # Encode element
                try:
                    result = encoder_func(element_info["tensor"], model, element_info["config"])
                    encoded_features[proc_name][element_name] = result
                except Exception as e:
                    logger.error(f"Error encoding {element_name} with {proc_name}: {e}")
                    # Continue with other elements
                    continue
        
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
        
        # For each output tensor defined in tensor_combinations
        for output_name, processor_list in tensor_combinations.items():
            processor_results = {}
            
            # For each processor contributing to this output
            for proc_name in processor_list:
                # Skip if processor not in encoded features
                if proc_name not in encoded_features:
                    logger.warning(f"Processor {proc_name} specified in tensor_combinations but not in encoded features")
                    continue
                    
                # Skip if no combiner registered for this processor
                if proc_name not in COMBINER_REGISTRY:
                    logger.warning(f"No combiner registered for processor type: {proc_name}")
                    continue
                
                # Get features for this processor
                processor_features = encoded_features[proc_name]
                if not processor_features:
                    logger.warning(f"No features for processor {proc_name}")
                    continue
                
                # Get combiner function
                combiner_func = COMBINER_REGISTRY[proc_name]
                
                # Combine features using the registered combiner
                try:
                    result = combiner_func(processor_features)
                    if result is not None:
                        processor_results[proc_name] = result
                except Exception as e:
                    logger.error(f"Error combining features for {proc_name}: {e}")
                    # Continue with other processors
                    continue
            
            # Create output tensor based on number of processors
            if len(processor_results) == 1:
                # Only one processor, use its result directly
                proc_name = list(processor_results.keys())[0]
                combined_tensors[output_name] = processor_results[proc_name]
            elif len(processor_results) > 1:
                # Multiple processors, concatenate along channel dimension
                try:
                    tensors = list(processor_results.values())
                    combined_tensors[output_name] = torch.cat(tensors, dim=1)  # Channel dimension
                except Exception as e:
                    logger.error(f"Error concatenating results for {output_name}: {e}")
                    logger.error(f"Tensor shapes: {[t.shape for t in list(processor_results.values())]}")
                    # Skip this output
                    continue
            else:
                # No results, log warning
                logger.warning(f"No valid results for output {output_name}")
        
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
        # Process inputs
        text_embeddings = batch.get("text_embeddings")
        video_latents = batch.get("latents")
        latents_mean = batch.get("latents_mean", None)
        latents_std = batch.get("latents_std", None)
        
        # Get preprocessed elements and configs
        preprocessed_elements = batch.get("preprocessed_elements", {})
        
        # Get tensor_combinations from batch (provided by dataset)
        tensor_combinations = batch.get("tensor_combinations")
        if tensor_combinations is None:
            raise ValueError(
                "No tensor_combinations found in batch data. "
                "E2V training requires explicit tensor_combinations configuration."
            )
        
        # Validate tensor combinations before encoding
        validate_tensor_combinations(tensor_combinations)
        logger.debug(f"Using tensor_combinations: {tensor_combinations}")
        
        # Process preprocessed elements through models
        encoded_features = self._encode_elements(preprocessed_elements)
        
        # Combine features according to tensor_combinations
        combined_tensors = self._combine_features(encoded_features, tensor_combinations)
        
        # Find control_latents (required for model)
        control_latents = find_tensor_by_key_pattern(combined_tensors, "condition_latents")
        if control_latents is None:
            raise ValueError(
                "No 'condition_latents' found in combined tensors. "
                "Please check your tensor_combinations configuration."
            )
        
        # Find clip_embeddings (optional)
        clip_embeddings = find_tensor_by_key_pattern(combined_tensors, "encoder_hidden_states") or \
                        find_tensor_by_key_pattern(combined_tensors, "clip")
        
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
        
        # Forward through model specification
        loss = self.model_specification.forward(
            transformer=self.transformer,
            condition_model_conditions=condition_model_conditions,
            latent_model_conditions=latent_model_conditions,
            sigmas=sigmas,
            generator=generator,
        )
        
        return loss

    def _update_parameters(self):
        """Update model parameters with the optimizer."""
        # Clip gradients
        if self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
            # Separate logic for FSDP
            if self.state.parallel_backend.is_fsdp:
                self.state.parallel_backend.clip_grad_norm_(self.args.clip_grad_norm)
            else:
                # Otherwise we do it on our trainable modules
                torch.nn.utils.clip_grad_norm_(
                    parameters=[p for p in self.transformer.parameters() if p.requires_grad],
                    max_norm=self.args.clip_grad_norm,
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

    def _validate(self):
        """Run validation."""
        if self.validation_dataloader is None:
            return
        
        logger.info("Running validation")
        parallel_backend = self.state.parallel_backend
        train_state = self.state.train_state
        
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
                        "step": train_state.global_step,
                    }
                    
                    # Log to trackers
                    parallel_backend.log_metrics(metrics, train_state.global_step)
                
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

    def _upload_to_hub(self):
        """Upload the final model to Hugging Face Hub."""
        if not self.state.parallel_backend.is_main_process:
            return
        
        logger.info("Uploading model to Hugging Face Hub")
        
        hub_model_id = self.args.hub_model_id or Path(self.args.output_dir).name
        
        repo_id = None
        if "/" in hub_model_id:
            repo_id = hub_model_id
        else:
            # Get organization name
            repo_id = f"{self.args.hub_organization}/{hub_model_id}" if self.args.hub_organization else hub_model_id
        
        # Create repo
        create_repo(repo_id, private=self.args.hub_private, token=self.args.hub_token, exist_ok=True)
        
        # Upload folder contents
        upload_folder(
            folder_path=self.args.output_dir,
            repo_id=repo_id,
            commit_message=f"Upload model {hub_model_id}",
            token=self.args.hub_token,
        )

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