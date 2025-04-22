"""Utility functions for E2V trainer."""
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from finetrainers.logging import get_logger

logger = get_logger()


def group_by_resolution(items: List[Tuple[Any, torch.Tensor]], 
                       batch_size: int = 1):
    """Group items by tensor resolution for efficient batching.
    
    Args:
        items: List of (sample, tensor) tuples
        batch_size: Maximum batch size
        
    Returns:
        List of batched items with similar resolutions
    """
    # Group items by shape
    shape_groups = {}
    for sample, tensor in items:
        # Use tensor shape as dictionary key
        shape_key = tuple(tensor.shape)
        if shape_key not in shape_groups:
            shape_groups[shape_key] = []
        shape_groups[shape_key].append((sample, tensor))

    # Create batches from each shape group
    batched_groups = []
    for shape, group in shape_groups.items():
        # Split into batches of batch_size
        for i in range(0, len(group), batch_size):
            batch = group[i:i+batch_size]
            batched_groups.append(batch)

    return batched_groups


def create_batch_from_tensors(tensor_items: List[torch.Tensor]):
    """Stack list of tensors into batch.
    
    Args:
        tensor_items: List of tensors with same shape
        
    Returns:
        Single batched tensor
    """
    return torch.stack(tensor_items, dim=0)


def validate_e2v_config(config: Dict[str, Any]) -> None:
    """Validate E2V configuration for correctness.
    
    Args:
        config: E2V configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate required sections
    required_keys = ["elements", "processors"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required E2V configuration keys: {missing_keys}. "
            f"E2V training requires explicit configuration for all required elements."
        )
    
    # Validate elements
    validate_elements_config(config.get("elements", []))
    
    # Validate processors
    validate_processors_config(config.get("processors", {}))
    
    # Validate tensor_combinations if present
    if "tensor_combinations" in config:
        validate_tensor_combinations(config["tensor_combinations"])


def validate_elements_config(elements: List[Dict[str, Any]]) -> None:
    """Validate elements configuration.
    
    Args:
        elements: List of element configurations
        
    Raises:
        ValueError: If elements configuration is invalid
    """
    if not elements:
        raise ValueError("E2V requires at least one element to be configured")
    
    for i, element in enumerate(elements):
        # Check required keys for each element
        if "name" not in element:
            raise ValueError(f"Element at index {i} is missing required 'name' field")
        
        if "suffixes" not in element:
            raise ValueError(f"Element '{element['name']}' is missing required 'suffixes' field")
        
        if not isinstance(element["suffixes"], list) or not element["suffixes"]:
            raise ValueError(
                f"Element '{element['name']}' has invalid 'suffixes'. "
                f"Expected non-empty list, got: {element['suffixes']}"
            )


def validate_processors_config(processors: Dict[str, Any]) -> None:
    """Validate processors configuration.
    
    Args:
        processors: Dictionary of processor configurations
        
    Raises:
        ValueError: If processors configuration is invalid
    """
    # Check if at least vae processor is present
    if "vae" not in processors:
        raise ValueError("E2V requires at least 'vae' processor to be configured")
    
    # Validate vae processor
    vae_config = processors.get("vae", {})
    if not isinstance(vae_config, dict):
        raise ValueError(f"Invalid 'vae' processor configuration. Expected dictionary, got: {type(vae_config)}")


def validate_tensor_combinations(tensor_combinations: Dict[str, List[str]]) -> None:
    """Validate tensor_combinations configuration.
    
    Args:
        tensor_combinations: Configuration dictionary for tensor combinations
        
    Raises:
        ValueError: If tensor_combinations is invalid
    """
    # Ensure tensor_combinations is a dictionary
    if not isinstance(tensor_combinations, dict) or not tensor_combinations:
        raise ValueError(
            f"Invalid tensor_combinations format. Expected non-empty dictionary, "
            f"got: {type(tensor_combinations)}"
        )
        
    # Validate each entry is a list of processor names
    for output_name, processor_list in tensor_combinations.items():
        if not isinstance(processor_list, list) or not processor_list:
            raise ValueError(
                f"Invalid tensor_combinations format for '{output_name}'. "
                f"Expected non-empty list of processor names, got: {processor_list}"
            )
    
    # Validate minimum required outputs
    required_outputs = ["condition_latents"]
    for required in required_outputs:
        if not any(required in key for key in tensor_combinations.keys()):
            raise ValueError(
                f"Required tensor output '{required}' not found in tensor_combinations. "
                f"Please check your tensor_combinations configuration."
            )


def is_processor_enabled(element_config: Dict[str, Any], proc_name: str) -> bool:
    """Check if a processor is enabled for an element.
    
    Args:
        element_config: Element configuration
        proc_name: Processor name to check
        
    Returns:
        True if processor is enabled, False otherwise
    """
    # Check if processor is explicitly disabled in element config
    if proc_name in element_config and element_config[proc_name] is False:
        return False
        
    # Check if processor is explicitly disabled in processors section
    if ("processors" in element_config and 
        proc_name in element_config["processors"] and 
        not element_config["processors"][proc_name]):
        return False
        
    return True


def get_processor_config(element_config: Dict[str, Any], proc_name: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract processor-specific configuration from element config.
    
    Args:
        element_config: Element configuration
        proc_name: Processor name to extract config for
        default_config: Default configuration to use as base
        
    Returns:
        Combined configuration dictionary
    """
    # Start with default config if provided
    config = {}
    if default_config:
        config.update(default_config)
        
    # Check 'processors' structure (preferred approach)
    if "processors" in element_config and proc_name in element_config["processors"]:
        if isinstance(element_config["processors"][proc_name], dict):
            config.update(element_config["processors"][proc_name])
            
    # Legacy: check direct keys
    elif proc_name in element_config:
        if isinstance(element_config[proc_name], dict):
            config.update(element_config[proc_name])
            
    return config


def find_tensor_by_key_pattern(combined_tensors: Dict[str, Any], pattern: str) -> Optional[Any]:
    """Find a tensor by key pattern in the combined tensors dictionary.
    
    Args:
        combined_tensors: Dictionary of combined tensors
        pattern: String pattern to search for in keys
        
    Returns:
        The found tensor, or None if not found
    """
    for key in combined_tensors:
        if pattern in key:
            return combined_tensors[key]
    return None