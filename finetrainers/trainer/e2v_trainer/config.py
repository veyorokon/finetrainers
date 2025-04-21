import argparse
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from finetrainers.trainer.config_utils import ConfigMixin


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


# Removed E2VType enum as it's redundant with configuration-driven approach


class FrameConditioningType(str, Enum):
    """Enum class for the frame conditioning types."""

    INDEX = "index"
    PREFIX = "prefix"
    RANDOM = "random"
    FIRST_AND_LAST = "first_and_last"
    FULL = "full"


class ElementConfig(ConfigMixin):
    """Configuration for a single element."""

    name: str
    suffixes: List[str]
    required: bool = False
    placeholder: Any = None  # Default value for placeholder elements
    vae: Dict[str, Any] = {"repeat": 1, "position": 0}
    clip: Union[Dict[str, Any], bool] = {"preprocess": "center_crop"}

    def validate_args(self, args):
        assert isinstance(self.name, str), "Element name must be a string"
        assert isinstance(self.suffixes, list), "Suffixes must be a list"
        assert all(isinstance(s, str) for s in self.suffixes), "All suffixes must be strings"
        
    def map_args(self, argparse_args, mapped_args):
        # No CLI args to map for element config
        pass
        
    def add_args(self, parser):
        # No CLI args to add for element config
        pass


class ProcessorConfig(ConfigMixin):
    """Base configuration for processors."""

    resolution: List[int]
    default_preprocess: str = "resize"
    batch_size: int = 16  # Default batch size for processing

    def validate_args(self, args):
        assert len(self.resolution) == 2, "Resolution must be [height, width]"
        assert self.batch_size > 0, "Batch size must be a positive integer"
        
    def map_args(self, argparse_args, mapped_args):
        # No CLI args to map for processor config
        pass
        
    def add_args(self, parser):
        # No CLI args to add for processor config
        pass


class VaeProcessorConfig(ProcessorConfig):
    """Configuration for VAE pathway."""

    combine: str = "before"
    frame_conditioning: str = FrameConditioningType.FULL
    frame_index: int = 0
    concatenate_mask: bool = True


class ClipProcessorConfig(ProcessorConfig):
    """Configuration for CLIP pathway."""

    default_preprocess: str = "center_crop"




class E2VConfig(ConfigMixin):
    """Base configuration for E2V training."""

    # e2v_type removed in favor of configuration-driven approach
    elements: List[ElementConfig] = []  # Default to empty list
    processors: Dict[str, Union[VaeProcessorConfig, ClipProcessorConfig]] = {}  # Default to empty dict
    tensor_combinations: Dict[str, List[str]] = {}  # Configuration for tensor combinations
    frame_conditioning_type: str = FrameConditioningType.FULL
    frame_conditioning_index: int = 0
    frame_conditioning_concatenate_mask: bool = True

    def validate_args(self, args):
        # Validate essential configuration properties
        if hasattr(self, 'elements') and self.elements:
            for elem in self.elements:
                assert isinstance(elem, ElementConfig), "Elements must be of type ElementConfig"
        
        if hasattr(self, 'processors') and self.processors:
            for proc_name, proc_config in self.processors.items():
                assert proc_name in ['vae', 'clip'], f"Unknown processor type: {proc_name}"
                assert isinstance(proc_config, (VaeProcessorConfig, ClipProcessorConfig)), \
                    f"Processor {proc_name} config must be of appropriate type"
                
        if hasattr(self, 'tensor_combinations') and self.tensor_combinations:
            for output_name, processor_list in self.tensor_combinations.items():
                assert isinstance(processor_list, list), f"Processor list for {output_name} must be a list"
                for proc_name in processor_list:
                    assert proc_name in ['vae', 'clip'], f"Unknown processor in tensor_combinations: {proc_name}"
    
    def map_args(self, argparse_args, mapped_args):
        # Map CLI args to this config
        self.frame_conditioning_type = getattr(argparse_args, "frame_conditioning_type", self.frame_conditioning_type)
        self.frame_conditioning_index = getattr(argparse_args, "frame_conditioning_index", self.frame_conditioning_index)
        self.frame_conditioning_concatenate_mask = getattr(
            argparse_args, "frame_conditioning_concatenate_mask", self.frame_conditioning_concatenate_mask
        )
    
    def add_args(self, parser):
        # Add E2V base args
        parser.add_argument(
            "--frame_conditioning_type",
            type=str,
            default=FrameConditioningType.FULL.value,
            choices=[x.value for x in FrameConditioningType.__members__.values()],
        )
        parser.add_argument("--frame_conditioning_index", type=int, default=0)
        parser.add_argument("--frame_conditioning_concatenate_mask", action="store_true")
        
    def map_from_json(self, json_config):
        """Map from JSON config to this class."""
        config = {}
        if "elements" in json_config:
            elements = []
            for element in json_config["elements"]:
                elements.append(ElementConfig(**element))
            config["elements"] = elements
        
        if "processors" in json_config:
            processors = {}
            if "vae" in json_config["processors"]:
                processors["vae"] = VaeProcessorConfig(**json_config["processors"]["vae"])
            if "clip" in json_config["processors"]:
                processors["clip"] = ClipProcessorConfig(**json_config["processors"]["clip"])
            config["processors"] = processors
        
        # Handle tensor_combinations if present
        if "tensor_combinations" in json_config:
            config["tensor_combinations"] = json_config["tensor_combinations"]
        
        # Copy other fields
        for key, value in json_config.items():
            if key not in ["elements", "processors", "tensor_combinations"]:
                config[key] = value
        
        # Update self with the new config
        for key, value in config.items():
            setattr(self, key, value)
        
        return self


class E2VLowRankConfig(E2VConfig):
    """Configuration for E2V low rank training."""

    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[str, List[str]] = (
        "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
    )
    train_qk_norm: bool = False

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=[
                "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
            ],
        )
        parser.add_argument("--train_qk_norm", action="store_true")

    def validate_args(self, args: "BaseArgs"):
        super().validate_args(args)
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."
        if isinstance(self.target_modules, str):
            # Single regex pattern is valid
            pass
        elif isinstance(self.target_modules, list):
            assert all(isinstance(m, str) for m in self.target_modules), "All target_modules entries must be strings"
        else:
            raise TypeError("target_modules must be a string or list of strings")
        
    def map_args(self, argparse_args, mapped_args):
        super().map_args(argparse_args, mapped_args)
        self.rank = getattr(argparse_args, "rank", self.rank)
        self.lora_alpha = getattr(argparse_args, "lora_alpha", self.lora_alpha)
        self.target_modules = getattr(argparse_args, "target_modules", self.target_modules)
        self.train_qk_norm = getattr(argparse_args, "train_qk_norm", self.train_qk_norm)


class E2VFullRankConfig(E2VConfig):
    """Configuration for E2V full rank training."""

    def add_args(self, parser: argparse.ArgumentParser):
        super().add_args(parser)
        
    def map_args(self, argparse_args, mapped_args):
        super().map_args(argparse_args, mapped_args)
        
    def validate_args(self, args: "BaseArgs"):
        super().validate_args(args)