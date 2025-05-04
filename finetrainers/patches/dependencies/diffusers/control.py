from contextlib import contextmanager
from typing import List, Union

import torch
from diffusers.hooks import HookRegistry, ModelHook

_CONTROL_CHANNEL_CONCATENATE_HOOK = "FINETRAINERS_CONTROL_CHANNEL_CONCATENATE_HOOK"


class ControlChannelConcatenateHook(ModelHook):
    """
    A hook that replaces the "hidden_states" tensor in the transformer's forward call.
    
    For A2, we need to use the first 16 channels for content and the next 20 channels
    for control. This hook handles that specialized replacement.
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
            # Get the tensor from args or kwargs
            if isinstance(input_name, int):
                original_tensor = args[input_name]
            else:
                original_tensor = kwargs[input_name]
            
            # Log tensor shapes for debugging
            self.logger.info(f"== Control Channel Hook ==")
            self.logger.info(f"Original tensor shape: {original_tensor.shape}")
            self.logger.info(f"Control tensor shape: {input_tensor.shape}")
            
            # Check channel sizes
            if original_tensor.shape[1] != 16:
                self.logger.warning(f"Expected 16 channels in original tensor, got {original_tensor.shape[1]}")
                
            if input_tensor.shape[1] != 20:
                self.logger.warning(f"Expected 20 channels in control tensor, got {input_tensor.shape[1]}")
            
            # For A2, we expect exactly a 16+20 channel structure
            # Direct replacement to maintain pipeline behavior
            result_tensor = torch.cat([original_tensor[:, :16], input_tensor], dim=1)
            
            self.logger.info(f"Final tensor shape: {result_tensor.shape}")
            
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
