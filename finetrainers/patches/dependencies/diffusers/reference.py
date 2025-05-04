from contextlib import contextmanager
from typing import List, Union

import torch
from diffusers.hooks import HookRegistry, ModelHook
from finetrainers.logging import get_logger

logger = get_logger()

_REFERENCE_CHANNEL_CONCATENATE_HOOK = "FINETRAINERS_REFERENCE_CHANNEL_CONCATENATE_HOOK"


class ReferenceChannelConcatenateHook(ModelHook):
    def __init__(self, input_names: List[str], inputs: List[torch.Tensor], dims: List[int], content_channels: int = 16):
        self.input_names = input_names
        self.inputs = inputs
        self.dims = dims
        self.content_channels = content_channels

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        for input_name, control_tensor, dim in zip(self.input_names, self.inputs, self.dims):
            original_tensor = args[input_name] if isinstance(input_name, int) else kwargs[input_name]
            
            # Extract just the first 16 content channels
            content_channels = original_tensor.narrow(dim, 0, self.content_channels)
            
            # Concatenate content with control channels for proper 36-channel format
            combined_tensor = torch.cat([content_channels, control_tensor], dim=dim)
            
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