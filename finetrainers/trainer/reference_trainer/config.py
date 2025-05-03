from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from finetrainers.trainer.config_utils import ConfigMixin
from finetrainers.trainer.control_trainer.config import (ControlLowRankConfig,
                                                         ControlType,
                                                         FrameConditioningType)


class ReferenceType(str, Enum):
    """Type of reference model to use."""

    A2 = "a2"
    CUSTOM = "custom"


@dataclass
class ReferenceConfig(ControlLowRankConfig):
    """Configuration for reference-based training.
    
    Extends ControlLowRankConfig to add reference-specific configuration.
    
    Configuration Priority:
    1. Command-line arguments explicitly provided by the user
    2. Values from the JSON configuration file (training.json)
    3. Default values defined in this class
    
    This ensures that JSON values take precedence over defaults, but users
    can still override specific values via command line when needed.
    """

    # Type of reference conditioning
    reference_type: ReferenceType = ReferenceType.A2
    
    # Reference image configuration (height, width format to match video_resolution_buckets)
    # All these values must be explicitly set in reference_config, no defaults
    vae_resolution: List[int] = None
    clip_resolution: List[int] = None
    reference_order: List[str] = None
    repeat_frames: List[int] = None
    
    # VAE combination method ('before' or 'after')
    # 'before': combine reference images before VAE encoding
    # 'after': encode each reference image separately and combine in latent space
    vae_combine: str = None
    
    # Reference patterns for finding reference images
    reference_suffixes: List[str] = None
    
    @classmethod
    def from_json(cls, json_path: str) -> "ReferenceConfig":
        """Create config with values from JSON file."""
        import json
        from finetrainers.logging import get_logger
        logger = get_logger()
        
        try:
            with open(json_path, "r") as file:
                dataset_configs = json.load(file)["datasets"]
                
            # Extract reference config from first dataset with it
            reference_config = {}
            for config in dataset_configs:
                if "reference_config" in config:
                    reference_config = config.get("reference_config", {})
                    logger.info(f"Found reference_config in JSON: {reference_config}")
                    break
                    
            # Create new instance with defaults
            config = cls()
            
            # Update with JSON values
            if "repeat_frames" in reference_config:
                config.repeat_frames = reference_config["repeat_frames"]
                logger.info(f"Loaded repeat_frames from JSON: {config.repeat_frames}")
            if "vae_resolution" in reference_config:
                config.vae_resolution = reference_config["vae_resolution"]
            if "clip_resolution" in reference_config:
                config.clip_resolution = reference_config["clip_resolution"]
            if "reference_order" in reference_config:
                config.reference_order = reference_config["reference_order"]
            if "reference_suffixes" in reference_config:
                config.reference_suffixes = reference_config["reference_suffixes"]
            if "vae_combine" in reference_config:
                config.vae_combine = reference_config["vae_combine"]
                logger.info(f"Loaded vae_combine from JSON: {config.vae_combine}")
                
            return config
        except Exception as e:
            logger.warning(f"Error loading reference config from JSON: {e}")
            return cls()  # Return default if JSON loading fails
    
    def validate(self) -> None:
        """Validate the configuration."""
        super().validate_args(None)  # Pass None as args since we're not using args here
        
        # Check that all required fields are set
        required_fields = [
            'vae_resolution', 'clip_resolution', 'reference_order', 
            'repeat_frames', 'reference_suffixes', 'vae_combine'
        ]
        
        for field in required_fields:
            if getattr(self, field) is None:
                raise ValueError(f"Required field '{field}' is not set in ReferenceConfig")
        
        # Validate field values
        if len(self.reference_order) < 1:
            raise ValueError("reference_order must have at least one entry")
            
        if len(self.repeat_frames) < len(self.reference_order):
            # Extend repeat_frames with 1s if needed
            self.repeat_frames.extend([1] * (len(self.reference_order) - len(self.repeat_frames)))
            
        if len(self.vae_resolution) != 2:
            raise ValueError("vae_resolution must be [height, width]")
            
        if len(self.clip_resolution) != 2:
            raise ValueError("clip_resolution must be [height, width]")
            
        if self.vae_combine not in ["before", "after"]:
            raise ValueError(f"vae_combine must be 'before' or 'after', got '{self.vae_combine}'")
    
    def add_args(self, parser):
        """Add reference-specific arguments to the parser."""
        import sys
        import json
        from finetrainers.logging import get_logger
        logger = get_logger()
        
        # First add control arguments
        super().add_args(parser)
        
        # Try to load JSON defaults if dataset_config is available
        json_defaults = {}
        for i, arg in enumerate(sys.argv):
            if arg == "--dataset_config" and i + 1 < len(sys.argv):
                dataset_config = sys.argv[i + 1]
                try:
                    with open(dataset_config, "r") as file:
                        datasets = json.load(file)["datasets"]
                        for config in datasets:
                            if "reference_config" in config:
                                json_defaults = config["reference_config"]
                                logger.info(f"Using JSON reference_config for CLI defaults: {json_defaults}")
                                break
                except Exception as e:
                    logger.warning(f"Failed to load reference config for CLI defaults: {e}")
                break
            elif arg.startswith("--dataset_config="):
                dataset_config = arg.split("=", 1)[1]
                try:
                    with open(dataset_config, "r") as file:
                        datasets = json.load(file)["datasets"]
                        for config in datasets:
                            if "reference_config" in config:
                                json_defaults = config["reference_config"]
                                logger.info(f"Using JSON reference_config for CLI defaults: {json_defaults}")
                                break
                except Exception as e:
                    logger.warning(f"Failed to load reference config for CLI defaults: {e}")
                break
                
        # Then add reference-specific arguments with JSON defaults if available
        parser.add_argument(
            "--reference_type", 
            type=str, 
            default=json_defaults.get("reference_type", ReferenceType.A2.value),
            choices=[x.value for x in ReferenceType.__members__.values()]
        )
        parser.add_argument(
            "--vae_resolution", 
            type=int, 
            nargs=2, 
            default=json_defaults.get("vae_resolution", [480, 854]), 
            help="[height, width]"
        )
        parser.add_argument(
            "--clip_resolution", 
            type=int, 
            nargs=2, 
            default=json_defaults.get("clip_resolution", [512, 512]), 
            help="[height, width]"
        )
        parser.add_argument(
            "--reference_order", 
            type=str, 
            nargs="+", 
            default=json_defaults.get("reference_order", ["object", "background"])
        )
        parser.add_argument(
            "--repeat_frames", 
            type=int, 
            nargs="+", 
            default=json_defaults.get("repeat_frames", [4, 1])
        )
        parser.add_argument(
            "--reference_suffixes", 
            type=str, 
            nargs="+", 
            default=json_defaults.get("reference_suffixes", ["_object", "_background"])
        )
        parser.add_argument(
            "--vae_combine",
            type=str,
            choices=["before", "after"],
            default=json_defaults.get("vae_combine"),  # No fallback default, require it in JSON
            help="Method for combining references: 'before' (combine images then VAE) or 'after' (VAE encode individually)"
        )
    
    def map_args(self, argparse_args, mapped_args):
        """Map arguments from argparse to the config."""
        # First map control arguments
        super().map_args(argparse_args, mapped_args)
        
        # Then map reference-specific arguments
        mapped_args.reference_type = argparse_args.reference_type
        mapped_args.vae_resolution = argparse_args.vae_resolution
        mapped_args.clip_resolution = argparse_args.clip_resolution
        mapped_args.reference_order = argparse_args.reference_order
        mapped_args.repeat_frames = argparse_args.repeat_frames
        mapped_args.reference_suffixes = argparse_args.reference_suffixes
        mapped_args.vae_combine = argparse_args.vae_combine
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        # Get control config dictionary
        result = super().to_dict()
        
        # Add reference-specific config
        result.update({
            "reference_type": self.reference_type,
            "vae_resolution": self.vae_resolution,
            "clip_resolution": self.clip_resolution,
            "reference_order": self.reference_order,
            "repeat_frames": self.repeat_frames,
            "reference_suffixes": self.reference_suffixes,
            "vae_combine": self.vae_combine
        })
        
        return result