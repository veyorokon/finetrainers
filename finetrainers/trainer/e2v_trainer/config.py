import argparse
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from finetrainers.trainer.control_trainer.config import ControlType, FrameConditioningType
from finetrainers.trainer.config_utils import ConfigMixin

if TYPE_CHECKING:
    from finetrainers.args import BaseArgs

class E2VConfig(ConfigMixin):
    """Configuration for E2V training, extending control configuration."""
    
    # Frame conditioning settings - directly mapped from control trainer
    frame_conditioning_type: str = FrameConditioningType.FULL
    frame_conditioning_index: int = 0
    frame_conditioning_concatenate_mask: bool = True
    
    # E2V specific configuration
    elements_config: Dict = {}  # Will be populated from JSON
    conditioning_config: Dict = {}  # Will be populated from JSON
    
    def add_args(self, parser: argparse.ArgumentParser):
        """Add E2V-specific arguments to parser."""
        # Add frame conditioning arguments for CLI consistency
        parser.add_argument(
            "--frame_conditioning_type",
            type=str,
            default=FrameConditioningType.FULL.value,
            choices=[x.value for x in FrameConditioningType.__members__.values()],
        )
        parser.add_argument("--frame_conditioning_index", type=int, default=0)
        parser.add_argument("--frame_conditioning_concatenate_mask", action="store_true")
    
    def validate_args(self, args: "BaseArgs"):
        """Validate E2V-specific arguments."""
        # Minimal validation
        pass
        
    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        """Map CLI arguments to config object."""
        # Map frame conditioning parameters
        mapped_args.frame_conditioning_type = argparse_args.frame_conditioning_type
        mapped_args.frame_conditioning_index = argparse_args.frame_conditioning_index
        mapped_args.frame_conditioning_concatenate_mask = argparse_args.frame_conditioning_concatenate_mask

class E2VLowRankConfig(E2VConfig):
    """Configuration for E2V low rank training."""
    
    # LoRA parameters
    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[str, List[str]] = None  # Will be set explicitly
    train_qk_norm: bool = False
    
    def add_args(self, parser: argparse.ArgumentParser):
        """Add E2V LoRA-specific arguments to parser."""
        super().add_args(parser)
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            required=True,
        )
        parser.add_argument("--train_qk_norm", action="store_true")
    
    def validate_args(self, args: "BaseArgs"):
        super().validate_args(args)
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."
        assert self.target_modules is not None, "target_modules must be specified for LoRA training"
    
    def map_args(self, argparse_args, mapped_args):
        super().map_args(argparse_args, mapped_args)
        mapped_args.rank = argparse_args.rank
        mapped_args.lora_alpha = argparse_args.lora_alpha
        mapped_args.target_modules = (
            argparse_args.target_modules[0] if len(argparse_args.target_modules) == 1 else argparse_args.target_modules
        )
        mapped_args.train_qk_norm = argparse_args.train_qk_norm


class E2VFullRankConfig(E2VConfig):
    """Configuration for E2V full rank training."""
    train_qk_norm: bool = False
    
    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--train_qk_norm", action="store_true")
    
    def map_args(self, argparse_args, mapped_args):
        super().map_args(argparse_args, mapped_args)
        mapped_args.train_qk_norm = argparse_args.train_qk_norm