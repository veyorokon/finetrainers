from contextlib import contextmanager
from typing import List, Union

import torch
from diffusers.hooks import HookRegistry, ModelHook

_CONTROL_CHANNEL_CONCATENATE_HOOK = "FINETRAINERS_CONTROL_CHANNEL_CONCATENATE_HOOK"


class ControlChannelConcatenateHook(ModelHook):
    """
    A hook that concatenates control tensors with the content latents for inference.
    This follows the A2 inference pattern where:
    1. Content latents (16 channels) come first in the combined tensor
    2. Control latents (20 channels) are appended to content
    3. The transformer operates on the combined tensor
    4. The scheduler only updates the content latents
    """
    def __init__(self, input_names: List[str], inputs: List[torch.Tensor], dims: List[int]):
        self.input_names = input_names
        self.inputs = inputs
        self.dims = dims
        
        # Import logging here to avoid circular imports
        from finetrainers.logging import get_logger
        self.logger = get_logger()

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        for input_name, input_tensor, dim in zip(self.input_names, self.inputs, self.dims):
            original_tensor = args[input_name] if isinstance(input_name, int) else kwargs[input_name]
            
            # Log tensor shapes for debugging
            self.logger.info(f"== Control Channel Hook ==")
            self.logger.info(f"Original tensor shape: {original_tensor.shape}")
            self.logger.info(f"Control tensor shape: {input_tensor.shape}")
            
            # Concatenate content with control along channel dimension
            result_tensor = torch.cat([original_tensor, input_tensor], dim=1)
            
            self.logger.info(f"Combined content with control, result shape: {result_tensor.shape}")
            
            if isinstance(input_name, int):
                args[input_name] = result_tensor
            else:
                kwargs[input_name] = result_tensor
                
        return args, kwargs


@contextmanager
def control_channel_concat(
    module: torch.nn.Module, input_names: List[Union[int, str]], inputs: List[torch.Tensor], dims: List[int]
):
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = ControlChannelConcatenateHook(input_names, inputs, dims)
    registry.register_hook(hook, _CONTROL_CHANNEL_CONCATENATE_HOOK)
    yield
    registry.remove_hook(_CONTROL_CHANNEL_CONCATENATE_HOOK, recurse=False)
