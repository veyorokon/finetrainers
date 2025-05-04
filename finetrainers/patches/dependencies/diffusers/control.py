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
            
            # Keep only the first 16 channels (content) from the original tensor
            # This ensures we don't mix content and control channels
            content_tensor = original_tensor[:, :16].clone()
            
            # Replace content tensor (should be 16 channels) with our input tensor
            result_tensor = torch.cat([content_tensor, input_tensor], dim=1)
            
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
