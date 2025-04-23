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
        
        # Initialize additional models
        self.image_encoder = None
        
        # Track what models are loaded
        self._clip_loaded = False
        
        # Add additional component names
        self._clip_component_names = ["image_encoder"]
        self._all_component_names.extend(self._clip_component_names)
    
    def _prepare_models(self) -> None:
        """Prepare models for training, extending parent with CLIP model."""
        # Call parent implementation first to load standard models
        super()._prepare_models()
        
        # Additionally load CLIP model
        logger.info("Loading image encoder for E2V training")
        condition_components = self.model_specification.load_condition_models()
        if "image_encoder" in condition_components:
            self.image_encoder = condition_components["image_encoder"]
            self._clip_loaded = True
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
    
    def _prepare_data(self, preprocessor, data_iterator):
        """Process data with optimized model coordination.
        
        This method:
        1. Collects samples into a buffer
        2. Processes text through text encoder
        3. Processes images through CLIP
        4. Processes videos through VAE
        5. Returns processed data for training
        """
        parallel_backend = self.state.parallel_backend
        
        # 1. Collect samples into buffer
        buffer_size = max(1, self.args.batch_size * self.args.gradient_accumulation_steps)
        collected_samples = []
        for _ in range(buffer_size):
            try:
                batch = next(data_iterator)
                # Handle batch format (list or single item)
                batch = batch[0] if isinstance(batch, list) else batch
                collected_samples.append(batch)
            except StopIteration:
                if not collected_samples:
                    # No samples available
                    logger.warning("Data iterator exhausted, no samples collected")
                    return None, None
                break
        
        # 2. Process all text data with text encoder
        if self.text_encoder is not None:
            self._move_components_to_device([self.text_encoder])
            collected_samples = self._process_text_batch(collected_samples)
            self._move_components_to_device([self.text_encoder], "cpu")
            utils.free_memory()
        
        # 3. Process all CLIP data with image encoder
        if self.image_encoder is not None:
            self._move_components_to_device([self.image_encoder])
            collected_samples = self._process_clip_batch(collected_samples)
            self._move_components_to_device([self.image_encoder], "cpu")
            utils.free_memory()
        
        # 4. Process all VAE data
        if self.vae is not None:
            self._move_components_to_device([self.vae])
            utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
            collected_samples = self._process_vae_batch(collected_samples)
            self._move_components_to_device([self.vae], "cpu")
            utils.free_memory()
        
        # 5. Process reference elements into conditioning tensors
        collected_samples = self._combine_conditions(collected_samples)
        
        # 6. Return to transformer for forward pass
        self._move_components_to_device([self.transformer])
        
        # Create iterators for the training loop
        condition_iterator = iter(collected_samples)
        latent_iterator = iter(collected_samples)
        
        return condition_iterator, latent_iterator
    
    def _process_text_batch(self, samples):
        """Process all text data through text encoder."""
        if not samples or self.text_encoder is None:
            return samples
        
        device = self.state.parallel_backend.device
        
        # Process each sample
        for i, sample in enumerate(samples):
            if "e2v_processed" not in sample or "text" not in sample["e2v_processed"]:
                continue
            
            # Get text data
            text_data = sample["e2v_processed"]["text"]
            if not text_data or "elements" not in text_data:
                continue
            
            # Process each text element
            for element_name, element_data in text_data["elements"].items():
                if "text" not in element_data:
                    continue
                
                text = element_data["text"]
                
                # Tokenize text
                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Encode text
                with torch.no_grad():
                    text_embeddings = self.text_encoder(**inputs)[0]
                
                # Store embeddings in sample
                if "encoder_hidden_states" not in sample:
                    sample["encoder_hidden_states"] = text_embeddings
        
        return samples
    
    def _process_clip_batch(self, samples):
        """Process all CLIP data through image encoder."""
        if not samples or self.image_encoder is None:
            return samples
        
        device = self.state.parallel_backend.device
        
        # Process each sample
        for i, sample in enumerate(samples):
            if "e2v_processed" not in sample or "clip" not in sample["e2v_processed"]:
                continue
            
            # Get clip data
            clip_data = sample["e2v_processed"]["clip"]
            if not clip_data or "elements" not in clip_data:
                continue
            
            # Process each element
            clip_features = []
            for element_name, element_data in sorted(
                clip_data["elements"].items(), 
                key=lambda x: x[1].get("position", 0)
            ):
                if "tensor" not in element_data:
                    continue
                
                # Process through CLIP model
                tensor = element_data["tensor"].to(device)
                
                with torch.no_grad():
                    # Apply normalization if needed
                    # Process through CLIP vision encoder
                    features = self.image_encoder(tensor, output_hidden_states=True)
                    # Use penultimate layer features
                    features = features.hidden_states[-2]
                
                clip_features.append(features)
            
            # Combine features if we have any
            if clip_features:
                # Concatenate along sequence dimension
                combined_features = torch.cat(clip_features, dim=1)
                
                # Store in sample
                sample["encoder_hidden_states_image"] = combined_features
        
        return samples
    
    def _process_vae_batch(self, samples):
        """Process all VAE data through VAE encoder."""
        if not samples or self.vae is None:
            return samples
        
        device = self.state.parallel_backend.device
        
        # Process target videos first
        for i, sample in enumerate(samples):
            if "video" not in sample:
                continue
            
            # Process video through VAE
            video = sample["video"].to(device)
            
            # Encode with VAE
            with torch.no_grad():
                latents = self.vae.encode(video).latent_dist.sample()
                latents = latents * 0.18215  # Scale factor for stable diffusion
            
            # Store latents in sample
            sample["latents"] = latents
        
        # Process reference elements
        for i, sample in enumerate(samples):
            if "e2v_processed" not in sample or "frame" not in sample["e2v_processed"]:
                continue
            
            # Get frame data
            frame_data = sample["e2v_processed"]["frame"]
            if not frame_data or "elements" not in frame_data:
                continue
            
            # Process each element
            reference_tensors = []
            positions = []
            
            for element_name, element_data in sorted(
                frame_data["elements"].items(), 
                key=lambda x: x[1].get("position", 0)
            ):
                if "tensor" not in element_data:
                    continue
                
                # Get tensor and metadata
                tensor = element_data["tensor"].to(device)
                position = element_data.get("position", 0)
                repeat = element_data.get("repeat", 1)
                
                # Repeat frames if needed
                if repeat > 1 and len(tensor.shape) >= 5:
                    # Tensor shape should be [B, C, F, H, W]
                    # Repeat along frame dimension (dim=2)
                    frame = tensor
                    repeated = []
                    for f in range(frame.size(2)):
                        f_tensor = frame[:, :, f:f+1]
                        f_repeated = torch.cat([f_tensor] * repeat, dim=2)
                        repeated.append(f_repeated)
                    
                    tensor = torch.cat(repeated, dim=2)
                
                reference_tensors.append((position, tensor))
                positions.append(position)
            
            # Sort by position
            reference_tensors.sort(key=lambda x: x[0])
            
            # Extract just the tensors in position order
            tensors = [t for _, t in reference_tensors]
            
            if not tensors:
                continue
                
            # Concatenate along temporal dimension
            if len(tensors) > 1:
                # For multiple reference elements, concatenate along time dimension
                combined = torch.cat(tensors, dim=2)  # dim=2 is frames dimension
            else:
                combined = tensors[0]
            
            # Encode through VAE
            with torch.no_grad():
                encoded = self.vae.encode(combined).latent_dist.sample()
                encoded = encoded * 0.18215  # Scale factor
            
            # Get conditioning parameters
            conditioning_config = frame_data.get("conditioning", {})
            frame_conditioning_type = conditioning_config.get("frame_conditioning_type", "full")
            concatenate_mask = conditioning_config.get("frame_conditioning_concatenate_mask", True)
            frame_conditioning_index = conditioning_config.get("frame_conditioning_index", 0)
            
            # Apply frame conditioning (from control_trainer)
            conditioned_latents = apply_frame_conditioning_on_latents(
                encoded,
                sample["latents"].shape[2],  # Target video frames
                channel_dim=1,
                frame_dim=2,
                frame_conditioning_type=frame_conditioning_type,
                frame_conditioning_index=frame_conditioning_index,
                concatenate_mask=concatenate_mask
            )
            
            # Store in sample
            sample["condition_latents"] = conditioned_latents
        
        return samples
    
    def _combine_conditions(self, samples):
        """Combine processed tensors based on configuration."""
        # This method would implement tensor_combinations logic
        # For now, we'll keep it simple and just ensure the required
        # tensors are available for the model
        
        return samples
    
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
    
    def _delete_components(self, component_names=None):
        """Delete components to free memory."""
        # Extend parent method to include image_encoder
        if component_names is None:
            component_names = self._all_component_names
        
        super()._delete_components(component_names)
    
    def _prepare_trainable_parameters(self) -> None:
        """Prepare trainable parameters based on training type."""
        # Handle E2V-specific training configurations
        if hasattr(self.args, 'training_type'):
            if self.args.training_type == TrainingType.E2V_LORA:
                logger.info("Setting up E2V with LoRA fine-tuning")
                utils.set_requires_grad([self.transformer], False)
                
                from peft import LoraConfig
                
                # Debug info on arguments
                logger.info(f"Using LoRA config with:")
                logger.info(f"  rank: {self.args.rank}")
                logger.info(f"  lora_alpha: {self.args.lora_alpha}")
                logger.info(f"  target_modules: {self.args.target_modules}")
                
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
        # Similar to parent implementation, but handle E2V-specific data
        if self.args.validation_dataset_file is None:
            return
        
        logger.info("Starting validation")
        
        # Load validation dataset
        parallel_backend = self.state.parallel_backend
        
        # Use the same dataset loading logic as in _prepare_dataset
        # but with ValidationE2VDataset wrapper
        dataset = data.ValidationDataset(self.args.validation_dataset_file)
        dataset = self.state.parallel_backend.prepare_dataset(dataset)
        
        # Load dataset config from args
        dataset_config = {"elements": self.args.elements_config, "conditioning": self.args.conditioning_config}
        
        # Wrap with E2V validation dataset
        dataset = ValidationE2VDataset(dataset, dataset_config, self.state.parallel_backend.device)
        
        # Rest of validation follows parent implementation
        # ...
        
        # We'll need to extend this with E2V-specific validation handling
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