import os
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import CLIPVisionModel

from finetrainers.logging import get_logger
from .control_specification import WanControlModelSpecification

logger = get_logger()


class WanE2VModelSpecification(WanControlModelSpecification):
    """Model specification for E2V training with Wan models.
    
    Extends WanControlModelSpecification to add support for the image_encoder
    needed for CLIP embedding pathway in E2V training.
    """
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        image_encoder_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        image_encoder_dtype: torch.dtype = torch.float32,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
            **kwargs,
        )
        self.image_encoder_id = image_encoder_id
        self.image_encoder_dtype = image_encoder_dtype

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        """Load condition models including CLIP image encoder for E2V training.
        
        Extends the parent method to additionally load the image_encoder required
        for the CLIP embedding pathway in E2V training.
        
        Returns:
            Dictionary of condition model components.
        """
        # First load the base condition models from the parent class
        components = super().load_condition_models()
        
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}
        
        # Add image encoder for CLIP pathway
        if self.image_encoder_id is not None:
            logger.info(f"Loading image encoder from {self.image_encoder_id}")
            image_encoder = CLIPVisionModel.from_pretrained(
                self.image_encoder_id, torch_dtype=self.image_encoder_dtype, **common_kwargs
            )
        else:
            image_encoder_path = str(Path(self.pretrained_model_name_or_path) / "image_encoder")
            logger.info(f"Loading image encoder from {image_encoder_path}")
            try:
                image_encoder = CLIPVisionModel.from_pretrained(
                    self.pretrained_model_name_or_path,
                    subfolder="image_encoder",
                    torch_dtype=self.image_encoder_dtype,
                    **common_kwargs
                )
                logger.info("Successfully loaded image encoder")
            except Exception as e:
                logger.warning(f"Failed to load image encoder: {e}")
                logger.warning("E2V training might not work correctly without the image encoder")
                image_encoder = None
        
        if image_encoder is not None:
            components["image_encoder"] = image_encoder
        
        return components
