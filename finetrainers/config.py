from enum import Enum
from typing import Type

from .models import ModelSpecification
from .models.wan import WanControlModelSpecification, WanModelSpecification


class ModelType(str, Enum):
    WAN = "wan"


class TrainingType(str, Enum):
    # SFT
    LORA = "lora"
    FULL_FINETUNE = "full-finetune"

    # Control
    CONTROL_LORA = "control-lora"
    CONTROL_FULL_FINETUNE = "control-full-finetune"
    
    # E2V
    E2V_LORA = "e2v-lora"
    E2V_FULL_FINETUNE = "e2v-full-finetune"


SUPPORTED_MODEL_CONFIGS = {
    # SFT, Control, and E2V
    ModelType.WAN: {
        TrainingType.LORA: WanModelSpecification,
        TrainingType.FULL_FINETUNE: WanModelSpecification,
        TrainingType.CONTROL_LORA: WanControlModelSpecification,
        TrainingType.CONTROL_FULL_FINETUNE: WanControlModelSpecification,
        TrainingType.E2V_LORA: WanControlModelSpecification,
        TrainingType.E2V_FULL_FINETUNE: WanControlModelSpecification,
    },
}


def _get_model_specifiction_cls(model_name: str, training_type: str) -> Type[ModelSpecification]:
    if model_name not in SUPPORTED_MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models are: {list(SUPPORTED_MODEL_CONFIGS.keys())}"
        )
    if training_type not in SUPPORTED_MODEL_CONFIGS[model_name]:
        raise ValueError(
            f"Training type {training_type} not supported for model {model_name}. Supported training types are: {list(SUPPORTED_MODEL_CONFIGS[model_name].keys())}"
        )
    return SUPPORTED_MODEL_CONFIGS[model_name][training_type]
