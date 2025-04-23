"""
Processors for Elements-to-Video (E2V) training.

These processors handle different conditioning pathways for E2V:
1. BasePathwayProcessor - Base class for all pathway processors
2. VAEPathwayProcessor - Handles spatial conditioning via VAE pathway
3. CLIPPathwayProcessor - Handles semantic conditioning via CLIP pathway
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPVisionModelWithProjection

from .base import ProcessorMixin


class BasePathwayProcessor(ProcessorMixin):
    """
    Base class for all E2V pathway processors.
    
    This processor serves as a foundation for pathway-specific processors
    in the E2V training framework.
    
    Args:
        output_names (List[str]): Names of the processor outputs
        input_names (Dict[str, Any], optional): Mapping of input names
    """

    def __init__(self, output_names: List[str] = None, input_names: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        
        self.output_names = output_names
        self.input_names = input_names
        
        assert len(output_names) >= 1, "At least one output name must be provided"

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Process inputs according to pathway requirements.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("BasePathwayProcessor::forward method should be implemented by subclasses")


class VAEPathwayProcessor(BasePathwayProcessor):
    """
    Processor for VAE pathway conditioning in E2V training.
    
    This processor handles spatial conditioning by preparing reference frames
    for the VAE encoder, including frame repetition and positioning.
    
    Args:
        output_names (List[str]): Names of the processor outputs
        input_names (Dict[str, Any], optional): Mapping of input names
    """

    def __init__(self, output_names: List[str] = None, input_names: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(output_names, input_names)
        
        assert len(output_names) == 1, "VAEPathwayProcessor requires exactly one output name"

    def forward(
        self,
        vae,
        frame_tensor: torch.Tensor,
        repeat: int = 1,
        scale_factor: float = 0.18215,
    ) -> Dict[str, Any]:
        """
        Process reference frames through the VAE encoder.
        
        Args:
            vae: The VAE model to use for encoding
            frame_tensor (torch.Tensor): Tensor containing reference frames [B, C, F, H, W]
            repeat (int, optional): Number of times to repeat each frame. Defaults to 1.
            scale_factor (float, optional): VAE latent scaling factor. Defaults to 0.18215.
            
        Returns:
            Dict containing VAE latents for the reference frames
        """
        device = vae.device
        
        # Move tensor to device if needed
        frame_tensor = frame_tensor.to(device)
        
        # Repeat frames if requested
        if repeat > 1:
            repeated = []
            for f in range(frame_tensor.size(2)):
                f_tensor = frame_tensor[:, :, f:f+1]
                f_repeated = torch.cat([f_tensor] * repeat, dim=2)
                repeated.append(f_repeated)
            
            frame_tensor = torch.cat(repeated, dim=2)
        
        # Encode through VAE
        with torch.no_grad():
            latents = vae.encode(frame_tensor).latent_dist.sample()
            latents = latents * scale_factor
        
        return {self.output_names[0]: latents}


class CLIPPathwayProcessor(BasePathwayProcessor):
    """
    Processor for CLIP pathway conditioning in E2V training.
    
    This processor handles semantic conditioning by extracting features
    from reference images using a CLIP vision model.
    
    Args:
        output_names (List[str]): Names of the processor outputs
        input_names (Dict[str, Any], optional): Mapping of input names
    """

    def __init__(self, output_names: List[str] = None, input_names: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(output_names, input_names)
        
        assert len(output_names) == 1, "CLIPPathwayProcessor requires exactly one output name"

    def forward(
        self,
        image_encoder: CLIPVisionModelWithProjection,
        image_tensor: torch.Tensor,
        use_penultimate_layer: bool = True,
    ) -> Dict[str, Any]:
        """
        Process images through the CLIP vision encoder.
        
        Args:
            image_encoder (CLIPVisionModelWithProjection): CLIP vision model
            image_tensor (torch.Tensor): Tensor containing images [B, C, H, W]
            use_penultimate_layer (bool, optional): Whether to use the penultimate
                hidden state instead of the final output. Defaults to True.
                
        Returns:
            Dict containing CLIP features for the images
        """
        device = image_encoder.device
        
        # Move tensor to device if needed
        image_tensor = image_tensor.to(device)
        
        # Process through CLIP vision encoder
        with torch.no_grad():
            features = image_encoder(image_tensor, output_hidden_states=use_penultimate_layer)
            
            # Use penultimate layer features if requested
            if use_penultimate_layer and features.hidden_states is not None:
                features = features.hidden_states[-2]
            else:
                features = features.last_hidden_state
        
        return {self.output_names[0]: features}