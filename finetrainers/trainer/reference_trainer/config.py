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
    """

    # Type of reference conditioning
    reference_type: ReferenceType = ReferenceType.A2
    
    # Reference image configuration (height, width format to match video_resolution_buckets)
    vae_resolution: List[int] = field(default_factory=lambda: [480, 854])
    clip_resolution: List[int] = field(default_factory=lambda: [512, 512])
    reference_order: List[str] = field(default_factory=lambda: ["object", "background"])
    repeat_frames: List[int] = field(default_factory=lambda: [4, 1])
    
    # Reference patterns for finding reference images
    reference_suffixes: List[str] = field(default_factory=lambda: ["_object", "_background"])
    
    def validate(self) -> None:
        """Validate the configuration."""
        super().validate_args(None)  # Pass None as args since we're not using args here
        
        if len(self.reference_order) < 1:
            raise ValueError("reference_order must have at least one entry")
            
        if len(self.repeat_frames) < len(self.reference_order):
            # Extend repeat_frames with 1s if needed
            self.repeat_frames.extend([1] * (len(self.reference_order) - len(self.repeat_frames)))
            
        if len(self.vae_resolution) != 2:
            raise ValueError("vae_resolution must be [height, width]")
            
        if len(self.clip_resolution) != 2:
            raise ValueError("clip_resolution must be [height, width]")
    
    def add_args(self, parser):
        """Add reference-specific arguments to the parser."""
        # First add control arguments
        super().add_args(parser)
        
        # Then add reference-specific arguments
        parser.add_argument("--reference_type", type=str, default=ReferenceType.A2.value,
                           choices=[x.value for x in ReferenceType.__members__.values()])
        parser.add_argument("--vae_resolution", type=int, nargs=2, default=[480, 854], help="[height, width]")
        parser.add_argument("--clip_resolution", type=int, nargs=2, default=[512, 512], help="[height, width]")
        parser.add_argument("--reference_order", type=str, nargs="+", default=["object", "background"])
        parser.add_argument("--repeat_frames", type=int, nargs="+", default=[4, 1])
        parser.add_argument("--reference_suffixes", type=str, nargs="+", default=["_object", "_background"])
    
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
            "reference_suffixes": self.reference_suffixes
        })
        
        return result