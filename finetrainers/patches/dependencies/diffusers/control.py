from contextlib import contextmanager
from typing import List, Union

import torch
from diffusers.hooks import HookRegistry, ModelHook

_CONTROL_CHANNEL_CONCATENATE_HOOK = "FINETRAINERS_CONTROL_CHANNEL_CONCATENATE_HOOK"


class ControlChannelConcatenateHook(ModelHook):
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
            self.logger.info(f"== Control Replacement Hook ==")
            self.logger.info(f"Original tensor shape: {original_tensor.shape}")
            self.logger.info(f"Input tensor shape: {input_tensor.shape}")
            
            # Replace the last 20 channels (4 mask + 16 conditioning) with our input tensor
            # For A2 model, we expect original to have 36 channels and input to have 20 channels
            # Keep first 16 channels (content) and replace last 20 (mask + conditioning)
            result_tensor = original_tensor.clone()
            # Replace the last 20 channels with our input tensor
            result_tensor[:, 16:, ...] = input_tensor
            
            self.logger.info(f"Replaced last 20 channels, result shape: {result_tensor.shape}")
        
            
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
