import json
import torch
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

if TYPE_CHECKING:
    from finetrainers.args import BaseArgs

from finetrainers import data, logging, utils as ft_utils
from finetrainers.trainer.control_trainer.trainer import ControlTrainer

from .config import E2VConfig, E2VFullRankConfig, E2VLowRankConfig, FrameConditioningType
from .data import IterableE2VDataset, ValidationE2VDataset
from .combiners import get_combiner, get_encoder
from .utils import group_by_resolution, create_batch_from_tensors

logger = logging.get_logger()


class E2VTrainer(ControlTrainer):
    """Elements-to-Video trainer that extends ControlTrainer with E2V-specific functionality."""

    def __init__(self, args: Union["BaseArgs", E2VFullRankConfig, E2VLowRankConfig], model_specification: Any) -> None:
        """Initialize E2V trainer, reusing ControlTrainer initialization."""
        # We need to ensure frame_conditioning attributes are present before calling super().__init__
        if not hasattr(args, 'frame_conditioning_type'):
            args.frame_conditioning_type = FrameConditioningType.FULL
        if not hasattr(args, 'frame_conditioning_index'):
            args.frame_conditioning_index = 0
        if not hasattr(args, 'frame_conditioning_concatenate_mask'):
            args.frame_conditioning_concatenate_mask = True
            
        super().__init__(args, model_specification)
        # Minimal additional initialization specific to E2V
        self.args = args
        self.image_encoder = None  # CLIP vision encoder

    def _prepare_models(self) -> None:
        """Prepare models, extending ControlTrainer's implementation."""
        super()._prepare_models()
        # Load CLIP vision encoder if needed
        if hasattr(self.model_specification, 'load_image_encoder'):
            self.image_encoder = self.model_specification.load_image_encoder()

    def _prepare_dataset(self) -> None:
        """Prepare dataset with E2V-specific configuration."""
        logger.info("Initializing dataset and dataloader for E2V training")

        with open(self.args.dataset_config, "r") as file:
            dataset_configs = json.load(file)["datasets"]
        logger.info(f"Training configured to use {len(dataset_configs)} datasets")

        datasets = []
        for config in dataset_configs:
            data_root = config.pop("data_root", None)
            dataset_file = config.pop("dataset_file", None)
            dataset_type = config.pop("dataset_type")
            caption_options = config.pop("caption_options", {})

            if data_root is not None and dataset_file is not None:
                raise ValueError("Both data_root and dataset_file cannot be provided in the same dataset config.")

            dataset_name_or_root = data_root or dataset_file
            dataset = data.initialize_dataset(
                dataset_name_or_root, dataset_type, streaming=True, infinite=True, _caption_options=caption_options
            )

            logger.info(f"Initialized dataset: {dataset_name_or_root}")
            dataset = self.state.parallel_backend.prepare_dataset(dataset)
            dataset = data.wrap_iterable_dataset_for_preprocessing(dataset, dataset_type, config)
            datasets.append(dataset)

        dataset = data.combine_datasets(datasets, buffer_size=self.args.dataset_shuffle_buffer_size, shuffle=True)
        
        # Create E2V-specific dataset wrapper with configuration
        dataset = IterableE2VDataset(dataset, config, self.state.parallel_backend.device)
        
        dataloader = self.state.parallel_backend.prepare_dataloader(
            dataset, batch_size=1, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.pin_memory
        )

        self.dataset = dataset
        self.dataloader = dataloader

    def _prepare_data(self, preprocessor, data_iterator):
        """Implement optimized model coordination pattern."""
        parallel_backend = self.state.parallel_backend
        device = parallel_backend.device
        
        logger.info("Using optimized model coordination for E2V data processing")
        
        # Collect samples to process
        collected_samples = []
        buffer_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        
        for _ in range(buffer_size):
            try:
                batch = next(data_iterator)
                batch = batch[0] if isinstance(batch, list) else batch
                collected_samples.append(batch)
            except StopIteration:
                # Handle dataset exhaustion
                data_iterator = iter(self.dataloader)
                batch = next(data_iterator)
                batch = batch[0] if isinstance(batch, list) else batch
                collected_samples.append(batch)
        
        # 1. Process text data
        if self.text_encoder is not None:
            self._move_components_to_device([self.text_encoder])
            collected_samples = self._process_text_batch(collected_samples)
            self._move_components_to_device([self.text_encoder], "cpu")
            ft_utils.free_memory()
        
        # 2. Process CLIP data
        if self.image_encoder is not None:
            self._move_components_to_device([self.image_encoder])
            collected_samples = self._process_clip_batch(collected_samples)
            self._move_components_to_device([self.image_encoder], "cpu")
            ft_utils.free_memory()
        
        # 3. Process VAE data
        if self.vae is not None:
            self._move_components_to_device([self.vae])
            ft_utils._enable_vae_memory_optimizations(self.vae, self.args.enable_slicing, self.args.enable_tiling)
            collected_samples = self._process_vae_batch(collected_samples)
            self._move_components_to_device([self.vae], "cpu")
            ft_utils.free_memory()
        
        # 4. Move transformer back to device
        self._move_components_to_device([self.transformer])
        
        # Create iterators from processed samples
        condition_iterator = iter(collected_samples)
        latent_iterator = iter(collected_samples)
        
        return condition_iterator, latent_iterator

    def _process_text_batch(self, collected_samples):
        """Process all text data through text encoder."""
        if self.text_encoder is None or self.tokenizer is None:
            return collected_samples
            
        device = self.state.parallel_backend.device
        
        # Group samples by caption length for efficient batching
        text_items = []
        for sample in collected_samples:
            if "caption" in sample:
                text_items.append((sample, sample["caption"]))
        
        # Group by token length (use first 10 chars as proxy for similar lengths)
        text_batches = []
        for i in range(0, len(text_items), self.args.batch_size):
            batch = text_items[i:i+self.args.batch_size]
            text_batches.append(batch)
        
        # Process each batch
        for batch in text_batches:
            samples = [item[0] for item in batch]
            captions = [item[1] for item in batch]
            
            # Tokenize and encode
            with torch.no_grad():
                batch_inputs = self.tokenizer(
                    captions, 
                    padding="max_length", 
                    max_length=self.tokenizer.model_max_length, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(device)
                
                batch_embeddings = self.text_encoder(batch_inputs)[0]
                
            # Store embeddings in samples
            for i, sample in enumerate(samples):
                sample["text_embeddings"] = batch_embeddings[i:i+1]  # Keep batch dimension
            
        return collected_samples

    def _process_clip_batch(self, collected_samples):
        """Process reference images through CLIP vision encoder."""
        if self.image_encoder is None:
            return collected_samples
            
        device = self.state.parallel_backend.device
        encode_clip = get_encoder("clip")
        
        # Collect all elements that need CLIP processing
        clip_items = []
        for sample_idx, sample in enumerate(collected_samples):
            if "preprocessed_elements" not in sample:
                continue
                
            clip_elements = sample.get("preprocessed_elements", {}).get("clip", {})
            if not clip_elements:
                continue
                
            # Process each element in this sample
            for element_name, element_data in clip_elements.items():
                tensor = element_data.get("tensor")
                if tensor is None:
                    continue
                    
                # Add to items list with metadata
                clip_items.append((
                    {
                        "sample_idx": sample_idx,
                        "element_name": element_name,
                        "config": element_data.get("config", {}),
                        "position": element_data.get("position", 0)
                    },
                    tensor
                ))
        
        # Group by resolution for efficient batching
        for batch in group_by_resolution(clip_items, self.args.batch_size):
            # Extract metadata and tensors
            metadatas = [item[0] for item in batch]
            tensors = [item[1] for item in batch]
            
            # Stack tensors into a single batch
            if len(tensors) > 1:
                batch_tensor = create_batch_from_tensors(tensors)
            else:
                batch_tensor = tensors[0]
            
            # Process through CLIP encoder
            with torch.no_grad():
                batch_tensor = batch_tensor.to(device)
                batch_features = encode_clip(batch_tensor, self.image_encoder)
            
            # Store features in respective samples
            for i, metadata in enumerate(metadatas):
                sample_idx = metadata["sample_idx"]
                element_name = metadata["element_name"]
                position = metadata["position"]
                
                # Create features structure if needed
                if "clip_features" not in collected_samples[sample_idx]:
                    collected_samples[sample_idx]["clip_features"] = {}
                
                # Extract this element's features from batch
                if len(batch_features) > 1:
                    features = {
                        "latents": batch_features[i:i+1],  # Keep batch dimension
                        "position": position
                    }
                else:
                    features = {
                        "latents": batch_features,
                        "position": position
                    }
                
                collected_samples[sample_idx]["clip_features"][element_name] = features
        
        # Apply combiners to processed features
        for sample in collected_samples:
            if "clip_features" not in sample:
                continue
                
            # Check tensor_combinations configuration
            tensor_combinations = sample.get("tensor_combinations", {})
            processor_configs = sample.get("processor_configs", {})
            
            # Combine CLIP features if needed
            if "encoder_hidden_states" in tensor_combinations and "clip" in tensor_combinations["encoder_hidden_states"]:
                # Get combiner and combine features
                combiner = get_combiner("clip")
                clip_config = processor_configs.get("clip", {})
                combined = combiner(sample["clip_features"], clip_config)
                
                if combined is not None:
                    sample["encoder_hidden_states_image"] = combined
        
        return collected_samples

    def _process_vae_batch(self, collected_samples):
        """Process video and reference data through VAE."""
        if self.vae is None:
            return collected_samples
            
        device = self.state.parallel_backend.device
        encode_vae = get_encoder("vae")
        
        # 1. Process target videos first
        video_items = []
        for sample_idx, sample in enumerate(collected_samples):
            if "video" in sample:
                video_items.append(({"sample_idx": sample_idx}, sample["video"]))
        
        # Process video batches
        for batch in group_by_resolution(video_items, self.args.batch_size):
            # Extract metadata and tensors
            metadatas = [item[0] for item in batch]
            tensors = [item[1] for item in batch]
            
            # Stack tensors into a single batch
            if len(tensors) > 1:
                batch_tensor = create_batch_from_tensors(tensors)
            else:
                batch_tensor = tensors[0]
            
            # Process through VAE
            with torch.no_grad():
                batch_tensor = batch_tensor.to(device)
                batch_latents = encode_vae(batch_tensor, self.vae)
            
            # Store in respective samples
            for i, metadata in enumerate(metadatas):
                sample_idx = metadata["sample_idx"]
                
                # Extract this sample's latents from batch
                if len(batch_latents) > 1:
                    latents = batch_latents["latents"][i:i+1]  # Keep batch dimension
                else:
                    latents = batch_latents["latents"]
                
                collected_samples[sample_idx]["latents"] = latents
        
        # 2. Process reference elements
        vae_items = []
        for sample_idx, sample in enumerate(collected_samples):
            if "preprocessed_elements" not in sample:
                continue
                
            vae_elements = sample.get("preprocessed_elements", {}).get("vae", {})
            if not vae_elements:
                continue
                
            # Process each element
            for element_name, element_data in vae_elements.items():
                tensor = element_data.get("tensor")
                if tensor is None:
                    continue
                
                # Handle repetition
                repeat = element_data.get("repeat", 1)
                if repeat > 1 and hasattr(tensor, "repeat"):
                    # Add frame dimension if needed
                    if len(tensor.shape) == 4:  # [B, C, H, W]
                        tensor = tensor.unsqueeze(2)  # [B, C, 1, H, W]
                    # Repeat frames
                    if len(tensor.shape) == 5:  # [B, C, F, H, W]
                        tensor = tensor.repeat(1, 1, repeat, 1, 1)
                
                # Add to items list
                vae_items.append((
                    {
                        "sample_idx": sample_idx,
                        "element_name": element_name,
                        "config": element_data.get("config", {}),
                        "position": element_data.get("position", 0)
                    },
                    tensor
                ))
        
        # Process VAE element batches
        for batch in group_by_resolution(vae_items, self.args.batch_size):
            # Extract metadata and tensors
            metadatas = [item[0] for item in batch]
            tensors = [item[1] for item in batch]
            
            # Stack tensors into a single batch
            if len(tensors) > 1:
                batch_tensor = create_batch_from_tensors(tensors)
            else:
                batch_tensor = tensors[0]
            
            # Process through VAE
            with torch.no_grad():
                batch_tensor = batch_tensor.to(device)
                batch_features = encode_vae(batch_tensor, self.vae)
            
            # Store in respective samples
            for i, metadata in enumerate(metadatas):
                sample_idx = metadata["sample_idx"]
                element_name = metadata["element_name"]
                position = metadata["position"]
                
                # Create features structure if needed
                if "vae_features" not in collected_samples[sample_idx]:
                    collected_samples[sample_idx]["vae_features"] = {}
                
                # Extract this element's features
                if len(batch_features) > 1:
                    features = {
                        "latents": batch_features["latents"][i:i+1],  # Keep batch dimension
                        "position": position
                    }
                else:
                    features = {
                        "latents": batch_features["latents"],
                        "position": position
                    }
                
                collected_samples[sample_idx]["vae_features"][element_name] = features
        
        # Apply combiners to processed features
        for sample in collected_samples:
            if "vae_features" not in sample:
                continue
                
            # Check tensor_combinations configuration
            tensor_combinations = sample.get("tensor_combinations", {})
            processor_configs = sample.get("processor_configs", {})
            
            # Combine VAE features if needed
            if "condition_latents" in tensor_combinations and "vae" in tensor_combinations["condition_latents"]:
                # Get combiner and combine features
                combiner = get_combiner("vae")
                vae_config = processor_configs.get("vae", {})
                combined = combiner(sample["vae_features"], vae_config)
                
                if combined is not None:
                    sample["condition_latents"] = combined
        
        return collected_samples