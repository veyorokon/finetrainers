"""
E2V Trainer - Extends ControlTrainer with Elements-to-Video capabilities.
"""
import json
import torch
import logging
from typing import Dict, List, Union, Any, Optional, Iterable, Tuple
from functools import partial

from finetrainers.config import TrainingType
from finetrainers import data, utils
from finetrainers.trainer.control_trainer.trainer import ControlTrainer
from finetrainers.trainer.control_trainer.data import apply_frame_conditioning_on_latents

from .data import IterableE2VDataset, ValidationE2VDataset
from .config import E2VFullRankConfig, E2VLowRankConfig

logger = logging.getLogger(__name__)

class E2VTrainer(ControlTrainer):
    """Elements-to-Video trainer that extends ControlTrainer with E2V-specific functionality."""

    def __init__(self, args, model_specification):
        """Initialize the E2V trainer.
        
        Args:
            args: Configuration options
            model_specification: Model specification object
        """
        # Ensure frame conditioning attributes are present before calling super().__init__
        if not hasattr(args, 'frame_conditioning_type'):
            args.frame_conditioning_type = "full"
        if not hasattr(args, 'frame_conditioning_index'):
            args.frame_conditioning_index = 0
        if not hasattr(args, 'frame_conditioning_concatenate_mask'):
            args.frame_conditioning_concatenate_mask = True
        
        # Store dataset configuration from JSON for later use
        if hasattr(args, 'dataset_config') and args.dataset_config:
            with open(args.dataset_config, "r") as f:
                data_config = json.load(f)
                if "datasets" in data_config and len(data_config["datasets"]) > 0:
                    args.elements_config = data_config["datasets"][0].get("elements", [])
                    args.conditioning_config = data_config["datasets"][0].get("conditioning", {})
        
        super().__init__(args, model_specification)
        
        # Initialize CLIP image encoder
        self.image_encoder = None
        
        # Add image encoder to component names
        self._all_component_names.append("image_encoder")
    
    def _prepare_models(self) -> None:
        """Prepare models for training, extending parent with CLIP model."""
        # Call parent implementation first to load standard models
        super()._prepare_models()
        
        # Additionally load CLIP model
        logger.info("Loading image encoder for E2V training")
        condition_components = self.model_specification.load_condition_models()
        if "image_encoder" in condition_components:
            self.image_encoder = condition_components["image_encoder"]
            logger.info("Successfully loaded image encoder")
        else:
            logger.warning("No image encoder found in model specification")
    
    def _prepare_dataset(self) -> None:
        """Prepare dataset for E2V training."""
        logger.info("Initializing dataset for E2V training")
        
        # Load dataset configuration
        with open(self.args.dataset_config, "r") as file:
            dataset_configs = json.load(file)["datasets"]
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")
        
        # Prepare dataset similar to parent class, but use our custom dataset wrapper
        datasets = []
        for config in dataset_configs:
            data_root = config.pop("data_root", None)
            dataset_file = config.pop("dataset_file", None)
            dataset_type = config.pop("dataset_type")
            caption_options = config.pop("caption_options", {})
            
            if data_root is not None and dataset_file is not None:
                raise ValueError("Both data_root and dataset_file cannot be provided in the same dataset config.")
            
            dataset_name_or_root = data_root or dataset_file
            dataset = data.initialize_dataset(
                dataset_name_or_root, dataset_type, streaming=True, infinite=True, _caption_options=caption_options
            )
            
            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
            datasets.append(dataset)
        
        combined_dataset = data.combine_datasets(datasets, 
                                               buffer_size=self.args.dataset_shuffle_buffer_size, 
                                               shuffle=True)
        
        # Use our custom E2V dataset wrapper
        dataset = IterableE2VDataset(
            combined_dataset, 
            dataset_configs[0],  # Pass first dataset config
            self.state.parallel_backend.device
        )
        
        dataloader = self.state.parallel_backend.prepare_dataloader(
            dataset, 
            batch_size=1, 
            num_workers=self.args.dataloader_num_workers, 
            pin_memory=self.args.pin_memory
        )
        
        self.dataset = dataset
        self.dataloader = dataloader
    
    # We'll use the parent implementation of _prepare_data
    
    # Process methods for handling the results of _prepare_data will be handled by
    # the model specification and parent class
    
    def _move_components_to_device(self, components=None, device=None):
        """Move model components to specified device."""
        # Extend parent method to include image_encoder
        if components is None:
            components = [
                self.tokenizer if hasattr(self.tokenizer, "to") else None,
                self.text_encoder,
                self.transformer,
                self.vae,
                self.image_encoder
            ]
            components = [c for c in components if c is not None]
        
        super()._move_components_to_device(components, device)
    
    def _prepare_trainable_parameters(self) -> None:
        """Prepare trainable parameters based on training type."""
        # Handle E2V-specific training configurations
        if hasattr(self.args, 'training_type'):
            if self.args.training_type == TrainingType.E2V_LORA:
                logger.info("Setting up E2V with LoRA fine-tuning")
                utils.set_requires_grad([self.transformer], False)
                
                from peft import LoraConfig
                
                # Configure LoRA
                lora_config = LoraConfig(
                    r=self.args.rank,
                    lora_alpha=self.args.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=self.args.target_modules,
                )
                
                # Add LoRA adapter
                self.transformer.add_adapter(lora_config)
                
                # Log trainable parameters
                trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
                logger.info(f"Number of trainable parameters: {trainable_params}")
                
                return
                
            elif self.args.training_type == TrainingType.E2V_FULL_FINETUNE:
                logger.info("Setting up E2V with full fine-tuning")
                utils.set_requires_grad([self.transformer], True)
                return
        
        # Fall back to parent implementation for other training types
        super()._prepare_trainable_parameters()
    
    def _prepare_checkpointing(self) -> None:
        """Set up checkpointing for the trainer."""
        # Use parent implementation with minor modifications
        parallel_backend = self.state.parallel_backend
        
        def save_model_hook(state_dict: Dict[str, Any]) -> None:
            state_dict = utils.get_unwrapped_model_state_dict(state_dict)
            if parallel_backend.is_main_process:
                if hasattr(self.args, 'training_type') and self.args.training_type == TrainingType.E2V_LORA:
                    from peft import get_peft_model_state_dict
                    state_dict = get_peft_model_state_dict(self.transformer, state_dict)
                    # Save LoRA weights
                    self.model_specification._save_lora_weights(
                        self.args.output_dir, state_dict, None, self.scheduler,
                        {"lora_config": json.dumps({
                            "r": self.args.rank,
                            "lora_alpha": self.args.lora_alpha,
                            "init_lora_weights": "gaussian",
                            "target_modules": self.args.target_modules
                        }, indent=4)}
                    )
                elif hasattr(self.args, 'training_type') and self.args.training_type == TrainingType.E2V_FULL_FINETUNE:
                    # Save full model
                    self.model_specification._save_model(
                        self.args.output_dir, self.transformer, state_dict, self.scheduler
                    )
                else:
                    # Fall back to parent method
                    super()._save_model_hook(state_dict)
            
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
    
    def _validate(self, step=None, final_validation=False) -> None:
        """Run validation with E2V-specific handling."""
        if self.args.validation_dataset_file is None:
            return
        
        logger.info("Starting validation")
        
        # Load validation dataset
        dataset = data.ValidationDataset(self.args.validation_dataset_file)
        dataset = self.state.parallel_backend.prepare_dataset(dataset)
        
        # Load dataset config from args
        dataset_config = {"elements": self.args.elements_config, "conditioning": self.args.conditioning_config}
        
        # Wrap with E2V validation dataset
        validation_dataset = ValidationE2VDataset(dataset, dataset_config, self.state.parallel_backend.device)
        
        # Store the dataset to be used by the parent implementation
        self.validation_dataset = validation_dataset
        
        # Use parent validation implementation
        super()._validate(step, final_validation)

    def _get_lora_target_modules(self):
        """Get LoRA target modules."""
        target_modules = getattr(self.args, "target_modules", None)
        if target_modules is None:
            return None
            
        if isinstance(target_modules, list):
            target_modules = list(target_modules)  # Make a copy
        
        # Add control injection layer
        if hasattr(self.model_specification, "control_injection_layer_name"):
            if isinstance(target_modules, list):
                target_modules.append(f"^{self.model_specification.control_injection_layer_name}$")
            elif isinstance(target_modules, str):
                target_modules = f"(^{self.model_specification.control_injection_layer_name}$)|({target_modules})"
                
        return target_modules