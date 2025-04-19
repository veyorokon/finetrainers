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


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs
    from finetrainers.models import ControlModelSpecification

ArgsType = Union["BaseArgs", E2VFullRankConfig, E2VLowRankConfig]

logger = logging.get_logger()


class E2VTrainer:
    # fmt: off
    _all_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "unet", "vae", "scheduler", "image_encoder"]
    _condition_component_names = ["tokenizer", "tokenizer_2", "tokenizer_3", "text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]
    _latent_component_names = ["vae"]
    _diffusion_component_names = ["transformer", "unet", "scheduler"]
    # fmt: on

    def __init__(self, args: ArgsType, model_specification: "ControlModelSpecification") -> None:
        self.args = args
        self.state = State()
        self.state.train_state = TrainState()

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

        self._init_distributed()
        self._init_config_options()

        # Perform any patches that might be necessary for training to work as expected
        patches.perform_patches_for_training(self.args, self.state.parallel_backend)

        self.model_specification = model_specification
        self._are_condition_models_loaded = False

        # Pass frame conditioning parameters to model specification
        model_specification._trainer_init(
            args.frame_conditioning_type, args.frame_conditioning_index, args.frame_conditioning_concatenate_mask
        )

    def run(self) -> None:
        try:
            self._prepare_models()
            self._prepare_trainable_parameters()
            self._prepare_for_training()
            self._prepare_dataset()
            self._prepare_checkpointing()
            self._train()
        except Exception as e:
            logger.exception(f"Error during training: {e}")
            raise
        finally:
            self._cleanup()

    def _init_distributed(self) -> None:
        # Set up distributed training
        if self.args.distributed_type is None:
            if self.args.deepspeed_config:
                self.args.distributed_type = "deepspeed"
            elif torch.cuda.device_count() > 1:
                self.args.distributed_type = "multi_gpu"

        # Create parallel state
        parallel_state = parallel.create_parallel_state(
            distributed_type=self.args.distributed_type,
            deepspeed_config_file=self.args.deepspeed_config,
            num_gpu_processes=self.args.num_processes,
            mixed_precision=self.args.mixed_precision,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            use_bf16=self.args.use_bf16,
            seed=self.args.seed,
            set_torch_seed=self.args.set_torch_seed,
            devices=self.args.devices_map,
            torch_compile=self.args.torch_compile,
            use_fp8=self.args.use_fp8,
            set_cudnn=self.args.set_cudnn,
        )
        self.state.parallel_backend = parallel_state
        self.state.is_local_main_process = parallel_state.is_local_main_process
        self.state.is_world_process_zero = parallel_state.is_world_process_zero

    def _init_config_options(self) -> None:
        # Set up configuration options
        self.state.gradient_accumulation_steps = self.args.gradient_accumulation_steps

        self.state.logging_nan_or_inf = self.args.logging_nan_or_inf
        self.state.allow_tf32 = self.args.allow_tf32
        if self.state.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if not self.args.train_batch_size:
            self.args.train_batch_size = 1

        if not self.args.eval_batch_size:
            self.args.eval_batch_size = 1

        if self.args.setup_torch_compile:
            # Reset compilation cache
            os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"

    def _prepare_models(self) -> None:
        # Load model components
        logger.info("Preparing models")
        
        # Load condition models
        self._load_condition_models()
        
        # Load latent models
        self._load_latent_models()
        
        # Load diffusion models with expanded patch embedding to handle control channels
        self._load_diffusion_models()

    def _load_condition_models(self) -> None:
        # Load text encoders and tokenizers
        logger.info("Loading condition models")
        
        components = self.model_specification.load_condition_models()
        
        for name, component in components.items():
            setattr(self, name, component)
        
        self._are_condition_models_loaded = True

    def _load_latent_models(self) -> None:
        # Load VAE
        logger.info("Loading latent models")
        
        components = self.model_specification.load_latent_models()
        
        for name, component in components.items():
            setattr(self, name, component)

    def _load_diffusion_models(self) -> None:
        # Load transformer with expanded patch embedding for control channels
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
        logger.info("Preparing trainable parameters")
        
        # For LoRA training
        if isinstance(self.args, E2VLowRankConfig):
            # Configure LoRA
            if not hasattr(self.transformer, "peft_config"):
                lora_config = LoraConfig(
                    r=self.args.rank,
                    lora_alpha=self.args.lora_alpha,
                    target_modules=self.args.target_modules,
                    init_lora_weights="gaussian",
                    lora_dropout=0.0,
                    bias="none",
                )
                
                # Convert string regex patterns to actual module names
                if isinstance(lora_config.target_modules, str) or (
                    isinstance(lora_config.target_modules, list) and len(lora_config.target_modules) == 1
                ):
                    if isinstance(lora_config.target_modules, str):
                        target_modules_pattern = lora_config.target_modules
                    else:
                        target_modules_pattern = lora_config.target_modules[0]
                    
                    import re
                    
                    filtered_modules = []
                    for name, _ in self.transformer.named_modules():
                        if re.search(target_modules_pattern, name):
                            filtered_modules.append(name)
                    
                    lora_config.target_modules = filtered_modules
                
                from peft import get_peft_model
                
                get_peft_model(self.transformer, lora_config)
                
                # Add QK norm if needed
                if self.args.train_qk_norm:
                    trainable_params = []
                    for name, param in self.transformer.named_parameters():
                        if "norm_q" in name or "norm_k" in name:
                            param.requires_grad = True
                            trainable_params.append(name)
                    
                    logger.info(f"Added {len(trainable_params)} QK norm layers to trainable parameters")
            
            # Set training modules
            self.trainable_modules = [self.transformer]
        
        # For full fine-tuning
        else:
            # Set all transformer parameters to trainable
            for param in self.transformer.parameters():
                param.requires_grad = True
            
            self.trainable_modules = [self.transformer]

    def _prepare_for_training(self) -> None:
        # Prepare optimizer, scheduler and other training components
        logger.info("Preparing for training")
        
        parameters = []
        for module in self.trainable_modules:
            parameters.extend(p for p in module.parameters() if p.requires_grad)
        
        # Initialize optimizer
        self.optimizer = optimizer.create_optimizer(
            self.args.lr, parameters, self.args.weight_decay, self.args.scale_lr, self.args.adam_beta1, self.args.adam_beta2
        )
        
        # Setup for distributed training
        if not isinstance(self.args, E2VFullRankConfig):
            # For LoRA, use standard prepare
            (
                self.transformer,
                self.optimizer,
            ) = self.state.parallel_backend.prepare_model_and_optimizer(self.transformer, self.optimizer)
        else:
            # For full fine-tuning, use specialized prepare
            (
                self.transformer,
                self.optimizer,
            ) = self.state.parallel_backend.prepare_model_optimizer_for_fsdp(self.transformer, self.optimizer)

        # Create LR scheduler
        scheduler_args = {
            "optimizer": self.optimizer,
            "num_warmup_steps": self.args.lr_warmup_steps,
            "num_training_steps": self.args.max_train_steps,
        }
        
        self.lr_scheduler = optimizer.create_scheduler(self.args.lr_scheduler, **scheduler_args)

    def _prepare_dataset(self) -> None:
        logger.info("Preparing datasets")

        # Process E2V configuration for dataset
        for ds_config in self.args.dataset_configs:
            # Add E2V configuration if not already present
            if "e2v_config" not in ds_config:
                ds_config["e2v_config"] = {
                    "e2v_type": self.args.e2v_type,
                    "elements": getattr(self.args, "elements", []),
                    "processors": getattr(self.args, "processors", {}),
                    "frame_conditioning_type": self.args.frame_conditioning_type,
                    "frame_conditioning_index": self.args.frame_conditioning_index,
                    "frame_conditioning_concatenate_mask": self.args.frame_conditioning_concatenate_mask,
                }
        
        # Create the training dataset
        train_dataset = data.create_dataset(self.args.dataset_configs, is_train=True)
        
        # Create E2V dataset wrapper
        self.train_dataset = IterableE2VDataset(
            train_dataset, 
            {
                "e2v_type": self.args.e2v_type,
                "elements": getattr(self.args, "elements", []),
                "processors": getattr(self.args, "processors", {}),
                "frame_conditioning_type": self.args.frame_conditioning_type,
                "frame_conditioning_index": self.args.frame_conditioning_index,
                "frame_conditioning_concatenate_mask": self.args.frame_conditioning_concatenate_mask,
            },
            device=self.state.parallel_backend.device,
            clip_processor=getattr(self, "image_encoder", None),
            vae=self.vae
        )
        
        # Setup data loader
        self.train_dataloader = data.create_dataloader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )
        
        # For validation
        if self.args.validation_configs:
            validation_dataset = data.create_dataset(self.args.validation_configs, is_train=False)
            
            self.validation_dataset = ValidationE2VDataset(
                validation_dataset,
                {
                    "e2v_type": self.args.e2v_type,
                    "elements": getattr(self.args, "elements", []),
                    "processors": getattr(self.args, "processors", {}),
                    "frame_conditioning_type": self.args.frame_conditioning_type,
                    "frame_conditioning_index": self.args.frame_conditioning_index,
                    "frame_conditioning_concatenate_mask": self.args.frame_conditioning_concatenate_mask,
                },
                device=self.state.parallel_backend.device,
                clip_processor=getattr(self, "image_encoder", None),
                vae=self.vae
            )
            
            self.validation_dataloader = data.create_dataloader(
                self.validation_dataset,
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.dataloader_num_workers,
            )
        else:
            self.validation_dataset = None
            self.validation_dataloader = None

    def _prepare_checkpointing(self) -> None:
        from finetrainers.trackers import CheckpointManager, WandbTracker
        
        checkpointing_config = {
            "output_dir": self.args.output_dir,
            "save_strategy": self.args.checkpoint_save_strategy,
            "save_steps": self.args.checkpoint_save_steps,
            "save_total_limit": self.args.checkpoint_save_total_limit,
            "save_on_cuda": self.args.checkpoint_save_on_cuda,
            "save_with_optimizer": self.args.checkpoint_save_with_optimizer,
            "model_parallel_backend": self.state.parallel_backend,
            "is_main_process": self.state.is_world_process_zero,
        }
        
        self.checkpointer = CheckpointManager(**checkpointing_config)
        
        # WandB Tracking
        if self.args.report_to_wandb and self.state.is_world_process_zero:
            tracker_args = {
                "project": self.args.wandb_project or "e2v-trainer",
                "name": self.args.run_name,
                "id": self.args.wandb_run_id,
                "resume": "auto" if self.args.resume_from_checkpoint else "never",
                "api_key": self.args.wandb_api_key,
                "entity": self.args.wandb_entity,
                "config": {**vars(self.args)},  # Logging all args
            }
            
            self.tracker = WandbTracker(**tracker_args)
        else:
            self.tracker = None

    def _train(self) -> None:
        logger.info("Starting training")
        
        train_dataloader = self.train_dataloader
        
        # Number of update steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.state.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
        
        total_batch_size = self.args.train_batch_size * self.state.parallel_backend.world_size * self.state.gradient_accumulation_steps
        logger.info(f"  Num examples = {len(train_dataloader)}")
        logger.info(f"  Num epochs = {num_train_epochs}")
        logger.info(f"  Batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.state.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        
        # Set initial values
        self.state.train_state.epoch = 0
        self.state.train_state.global_step = 0
        self.state.train_state.max_steps = self.args.max_train_steps
        
        # Resume from checkpoint if needed
        if self.args.resume_from_checkpoint:
            self._resume_from_checkpoint()
        
        progress_bar = tqdm(
            range(self.state.train_state.global_step, self.args.max_train_steps),
            disable=not self.state.is_local_main_process,
            desc="Training steps",
        )
        
        for epoch in range(self.state.train_state.epoch, num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                # Skip steps already performed
                if self.state.train_state.global_step > 0 and epoch == self.state.train_state.epoch and step < self.state.train_state.steps_in_epoch:
                    continue
                
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
                        self.state.train_state.global_step += 1
                        self.state.train_state.steps_in_epoch = step + 1
                        
                        if self.tracker is not None:
                            self.tracker.log({
                                "loss": loss.detach().item(),
                                "lr": self.lr_scheduler.get_last_lr()[0],
                                "step": self.state.train_state.global_step,
                                "epoch": epoch,
                            })
                        
                        # Run validation
                        if self.args.validation_steps > 0 and self.state.train_state.global_step % self.args.validation_steps == 0:
                            self._validate()
                        
                        # Create checkpoint
                        if self.checkpointer.should_save(self.state.train_state.global_step):
                            self._create_checkpoint()
                
                # Check if we've reached max steps
                if self.state.train_state.global_step >= self.args.max_train_steps:
                    break
            
            self.state.train_state.epoch = epoch + 1
            self.state.train_state.steps_in_epoch = 0
        
        # Make sure we create a final checkpoint
        if self.state.train_state.global_step != 0:
            self._create_checkpoint()
            
            # Upload to Hugging Face Hub if specified
            if self.args.push_to_hub and self.state.is_world_process_zero:
                self._upload_to_hub()
    
    def _forward_pass(self, batch):
        """Run forward pass with E2V conditioning."""
        # Process inputs
        text_embeddings = batch.get("text_embeddings")
        video_latents = batch.get("latents")
        latents_mean = batch.get("latents_mean", None)
        latents_std = batch.get("latents_std", None)
        
        # Get E2V specific conditioning
        e2v_vae_latents = batch.get("e2v_vae_latents")
        e2v_clip_embeddings = batch.get("e2v_clip_embeddings")
        
        # Generate random sigmas for flow matching
        generator = torch.Generator(device=self.state.parallel_backend.device).manual_seed(self.args.seed)
        batch_size = video_latents.shape[0]
        
        # Prepare batch for model
        latent_model_conditions = {
            "latents": video_latents,
            "control_latents": e2v_vae_latents,
        }
        
        if latents_mean is not None and latents_std is not None:
            latent_model_conditions["latents_mean"] = latents_mean
            latent_model_conditions["latents_std"] = latents_std
        
        # Condition model
        condition_model_conditions = {
            "encoder_hidden_states": text_embeddings,
        }
        
        # Add CLIP embeddings if available
        if e2v_clip_embeddings is not None:
            condition_model_conditions["encoder_hidden_states_image"] = e2v_clip_embeddings
        
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
        
        with torch.no_grad():
            for i, batch in enumerate(self.validation_dataloader):
                # Only process a few batches
                if i >= self.args.max_validation_batches:
                    break
                
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss = self._forward_pass(batch)
                
                if self.tracker is not None:
                    self.tracker.log({
                        "val_loss": loss.detach().item(),
                        "step": self.state.train_state.global_step,
                    })
                
                # Generate sample if requested
                if self.args.validation_generate_samples and i == 0:
                    self._generate_samples(batch)
    
    def _generate_samples(self, batch):
        """Generate video samples using the current model."""
        # This would be nice to have but is beyond the scope of this initial implementation
        # Would need to create a proper pipeline that supports E2V generation
        pass
    
    def _create_checkpoint(self):
        """Create a checkpoint of the current model."""
        if not self.state.is_world_process_zero:
            return
        
        logger.info(f"Creating checkpoint at step {self.state.train_state.global_step}")
        
        ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.train_state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save training state
        torch.save(self.state.train_state, os.path.join(ckpt_dir, "train_state.bin"))
        
        # Save transformer
        if isinstance(self.args, E2VLowRankConfig):
            # Save adapter weights for LoRA
            if self.state.parallel_backend.is_fsdp:
                # Special handling for FSDP+LoRA
                pass  # Implement this when needed
            else:
                adapter_weights = get_peft_model_state_dict(self.transformer)
                torch.save(adapter_weights, os.path.join(ckpt_dir, "adapter_model.bin"))
        else:
            # Save full transformer (non-LoRA)
            self.checkpointer.save(
                state_dict=self.transformer.state_dict(), 
                config=self.transformer.config, 
                component_name="transformer",
                save_dir=ckpt_dir
            )
        
        # Save optimizer and scheduler if requested
        if self.args.checkpoint_save_with_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.bin"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.bin"))
    
    def _resume_from_checkpoint(self):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint {self.args.resume_from_checkpoint}")
        
        ckpt_dir = None
        if self.args.resume_from_checkpoint != "latest":
            ckpt_dir = os.path.join(self.args.output_dir, self.args.resume_from_checkpoint)
        else:
            # Find latest checkpoint
            dirs = [
                os.path.join(self.args.output_dir, d)
                for d in os.listdir(self.args.output_dir)
                if os.path.isdir(os.path.join(self.args.output_dir, d)) and d.startswith("checkpoint-")
            ]
            dirs.sort(key=lambda d: int(d.split("-")[-1]))
            if len(dirs) > 0:
                ckpt_dir = dirs[-1]
            else:
                logger.warning("No checkpoints found, starting from scratch.")
                return
        
        # Load training state
        if os.path.exists(os.path.join(ckpt_dir, "train_state.bin")):
            self.state.train_state = torch.load(os.path.join(ckpt_dir, "train_state.bin"))
        
        # Load transformer
        if isinstance(self.args, E2VLowRankConfig):
            # Load adapter weights for LoRA
            if os.path.exists(os.path.join(ckpt_dir, "adapter_model.bin")):
                self.transformer.load_state_dict(torch.load(os.path.join(ckpt_dir, "adapter_model.bin")), strict=False)
        else:
            # Load full transformer
            if self.state.parallel_backend.is_fsdp:
                # Special handling for FSDP
                pass  # Implement this when needed
            else:
                load_lora_weights(
                    self.transformer,
                    ckpt_dir,
                )
        
        # Load optimizer and scheduler if possible
        if self.args.resume_optimizer_from_checkpoint:
            if os.path.exists(os.path.join(ckpt_dir, "optimizer.bin")):
                self.optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer.bin")))
            if os.path.exists(os.path.join(ckpt_dir, "scheduler.bin")):
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(ckpt_dir, "scheduler.bin")))
    
    def _upload_to_hub(self):
        """Upload the final model to Hugging Face Hub."""
        if not self.state.is_world_process_zero:
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
        # For LoRA adapters, only upload the relevant files
        upload_path = self.args.output_dir
        upload_folder(
            folder_path=upload_path,
            repo_id=repo_id,
            commit_message=f"Upload model {hub_model_id}",
            token=self.args.hub_token,
        )
    
    def _cleanup(self):
        """Clean up resources after training."""
        if self.tracker is not None:
            self.tracker.finish()