from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from finetrainers.trainer.config_utils import ConfigBase
from finetrainers.trainer.control_trainer.config import ControlConfig, ControlType, FrameConditioningType


class ReferenceType(str, Enum):
    """Type of reference model to use."""

    A2 = "a2"
    CUSTOM = "custom"


@dataclass
class ReferenceConfig(ControlConfig):
    """Configuration for reference-based training.
    
    Extends ControlConfig to add reference-specific configuration.
    """

    # Type of reference conditioning
    reference_type: ReferenceType = ReferenceType.A2
    
    # Reference image configuration
    vae_resolution: List[int] = field(default_factory=lambda: [854, 480])
    clip_resolution: List[int] = field(default_factory=lambda: [512, 512])
    reference_order: List[str] = field(default_factory=lambda: ["object", "background"])
    repeat_frames: List[int] = field(default_factory=lambda: [1, 4])
    
    # Reference patterns for finding reference images
    reference_suffixes: List[str] = field(default_factory=lambda: ["_object", "_background"])
    
    def validate(self) -> None:
        """Validate the configuration."""
        super().validate()
        
        if len(self.reference_order) < 1:
            raise ValueError("reference_order must have at least one entry")
            
        if len(self.repeat_frames) < len(self.reference_order):
            # Extend repeat_frames with 1s if needed
            self.repeat_frames.extend([1] * (len(self.reference_order) - len(self.repeat_frames)))
            
        if len(self.vae_resolution) != 2:
            raise ValueError("vae_resolution must be [width, height]")
            
        if len(self.clip_resolution) != 2:
            raise ValueError("clip_resolution must be [width, height]")