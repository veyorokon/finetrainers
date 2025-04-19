import argparse
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from finetrainers.trainer.config_utils import ConfigMixin


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


class E2VType(str, Enum):
    """Enum class for the E2V processing types."""

    VAE = "vae"
    CLIP = "clip"
    DUAL = "dual"  # Both pathways enabled


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
    vae: Dict[str, Any] = {"repeat": 1, "position": 0}
    clip: Union[Dict[str, Any], bool] = {"preprocess": "center_crop"}

    def validate_args(self, args):
        assert isinstance(self.name, str), "Element name must be a string"
        assert isinstance(self.suffixes, list), "Suffixes must be a list"
        assert all(isinstance(s, str) for s in self.suffixes), "All suffixes must be strings"


class ProcessorConfig(ConfigMixin):
    """Base configuration for processors."""

    resolution: List[int]
    default_preprocess: str = "resize"

    def validate_args(self, args):
        assert len(self.resolution) == 2, "Resolution must be [height, width]"


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

    e2v_type: str = E2VType.DUAL
    elements: List[ElementConfig]
    processors: Dict[str, Union[VaeProcessorConfig, ClipProcessorConfig]]
    frame_conditioning_type: str = FrameConditioningType.FULL
    frame_conditioning_index: int = 0
    frame_conditioning_concatenate_mask: bool = True

    def validate_args(self, args):
        assert self.e2v_type in E2VType.__members__.values(), f"Invalid E2V type: {self.e2v_type}"
        assert len(self.elements) > 0, "At least one element must be specified"
        assert "vae" in self.processors, "VAE processor configuration is required"
        if self.e2v_type in [E2VType.CLIP, E2VType.DUAL]:
            assert "clip" in self.processors, "CLIP processor configuration is required"
        
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
        
        # Copy other fields
        for key, value in json_config.items():
            if key not in ["elements", "processors"]:
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
        parser.add_argument(
            "--e2v_type",
            type=str,
            default=E2VType.DUAL.value,
            choices=[x.value for x in E2VType.__members__.values()],
        )
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
        parser.add_argument(
            "--frame_conditioning_type",
            type=str,
            default=FrameConditioningType.FULL.value,
            choices=[x.value for x in FrameConditioningType.__members__.values()],
        )
        parser.add_argument("--frame_conditioning_index", type=int, default=0)
        parser.add_argument("--frame_conditioning_concatenate_mask", action="store_true")

    def validate_args(self, args: "BaseArgs"):
        super().validate_args(args)
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."


class E2VFullRankConfig(E2VConfig):
    """Configuration for E2V full rank training."""

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--e2v_type",
            type=str,
            default=E2VType.DUAL.value,
            choices=[x.value for x in E2VType.__members__.values()],
        )
        parser.add_argument(
            "--frame_conditioning_type",
            type=str,
            default=FrameConditioningType.FULL.value,
            choices=[x.value for x in FrameConditioningType.__members__.values()],
        )
        parser.add_argument("--frame_conditioning_index", type=int, default=0)
        parser.add_argument("--frame_conditioning_concatenate_mask", action="store_true")