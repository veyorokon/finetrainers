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
            self.logger.info(f"== Control Concatenation Hook ==")
            self.logger.info(f"Original tensor shape: {original_tensor.shape}")
            self.logger.info(f"Input tensor shape: {input_tensor.shape}")
            self.logger.info(f"Concatenation dimension: {dim}")
            
            # Check if tensor shapes match except in the concat dimension
            if original_tensor.ndim != input_tensor.ndim:
                self.logger.error(f"Tensor dimensions don't match: {original_tensor.ndim} vs {input_tensor.ndim}")
            
            for i in range(original_tensor.ndim):
                if i != dim and original_tensor.shape[i] != input_tensor.shape[i]:
                    self.logger.error(f"Mismatch in dimension {i}: {original_tensor.shape[i]} vs {input_tensor.shape[i]}")
            
            # Proceed with concatenation
            control_tensor = torch.cat([original_tensor, input_tensor], dim=dim)
            self.logger.info(f"Result tensor shape: {control_tensor.shape}")
            
            if isinstance(input_name, int):
                args[input_name] = control_tensor
            else:
                kwargs[input_name] = control_tensor
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
