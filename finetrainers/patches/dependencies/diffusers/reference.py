from contextlib import contextmanager
from functools import wraps
from typing import Any, List, Union

import torch
from diffusers.hooks import HookRegistry, ModelHook

from finetrainers.logging import get_logger

logger = get_logger()

_REFERENCE_CHANNEL_CONCATENATE_HOOK = "FINETRAINERS_REFERENCE_CHANNEL_CONCATENATE_HOOK"
_SCHEDULER_STEP_PATCH_HOOK = "FINETRAINERS_SCHEDULER_STEP_PATCH_HOOK"


class ReferenceChannelConcatenateHook(ModelHook):
    def __init__(self, input_names: List[str], inputs: List[torch.Tensor], dims: List[int], content_channels: int = 16):
        self.input_names = input_names
        self.inputs = inputs
        self.dims = dims
        self.content_channels = content_channels
        self.content_tensor = None

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        for input_name, control_tensor, dim in zip(self.input_names, self.inputs, self.dims):
            original_tensor = args[input_name] if isinstance(input_name, int) else kwargs[input_name]
            
            # Extract just the first 16 content channels
            content_channels = original_tensor.narrow(dim, 0, self.content_channels)
            
            # Store content tensor for the scheduler patch
            self.content_tensor = content_channels.clone()
            
            # Concatenate content with control channels for proper 36-channel format
            combined_tensor = torch.cat([content_channels, control_tensor], dim=dim)

            # Debug visualization of latent channels - only if enabled
            import os
            if os.environ.get("REFERENCE_DEBUG_LATENTS") == "1":
                from finetrainers.utils import create_channel_frame_grid

                # Create a unique identifier for this validation
                val_id = os.environ.get("REFERENCE_DEBUG_VAL_ID", "0")
                # Save to debug directory
                output_dir = os.path.join("debug_latents", f"validation_{val_id}")
                # Convert to float32 before visualization to avoid bfloat16 issues
                control_latents_float = combined_tensor.to(torch.float32)
                # Create channel×frame grid visualization only
                create_channel_frame_grid(
                    control_latents_float,
                    output_dir,
                    filename=f"validation_latent_grid_{val_id}.png",
                    group_sizes=[4, 16]
                )
                
                # Increment counter
                next_val = int(val_id) + 1
                os.environ["REFERENCE_DEBUG_VAL_ID"] = str(next_val)
            
            # Log the shapes for debugging
            logger.info(f"Reference hook: original={original_tensor.shape}, " +
                       f"content={content_channels.shape}, control={control_tensor.shape}, " +
                       f"combined={combined_tensor.shape}")
            
            # Update the tensor in args or kwargs
            if isinstance(input_name, int):
                args[input_name] = combined_tensor
            else:
                kwargs[input_name] = combined_tensor
                
        return args, kwargs


@contextmanager
def reference_channel_concat(
    module: torch.nn.Module, input_names: List[Union[int, str]], inputs: List[torch.Tensor], 
    dims: List[int], content_channels: int = 16
):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = ReferenceChannelConcatenateHook(input_names, inputs, dims, content_channels)
    registry.register_hook(hook, _REFERENCE_CHANNEL_CONCATENATE_HOOK)
    yield
    registry.remove_hook(_REFERENCE_CHANNEL_CONCATENATE_HOOK, recurse=False)


class SchedulerStepPatch(ModelHook):
    def __init__(self, scheduler, content_latents):
        self.scheduler = scheduler
        self.content_latents = content_latents
        self.original_step = scheduler.step
        
    def pre_forward(self, module, *args, **kwargs):
        # This won't be used, we're patching the step method directly
        return args, kwargs


@contextmanager
def scheduler_step_patch(scheduler, latents):
    """
    Patch the scheduler step method to use only the first 16 channels of latents.
    
    Args:
        scheduler: The scheduler instance to patch
        latents: The 36-channel latents (we'll extract first 16 channels)
    """
    original_step = scheduler.step
    
    @wraps(original_step)
    def patched_step(model_output, timestep, sample, *args, **kwargs):
        # Extract just the first 16 channels from the 36-channel latents
        content_latents = sample.narrow(1, 0, 16)
        logger.info(f"Scheduler patch: using first 16 channels from latents shape {sample.shape} → {content_latents.shape}")
        return original_step(model_output, timestep, content_latents, *args, **kwargs)
    
    try:
        scheduler.step = patched_step
        yield
    finally:
        scheduler.step = original_step