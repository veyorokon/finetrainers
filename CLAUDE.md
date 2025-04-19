# Project Goals and Context

## Overview
We're developing A2 training functionality within the finetrainers framework. A2 is not a separate model but rather a specialized training approach built on top of the Wan video diffusion model for "Elements-to-Video" (E2V) tasks. The A2 folder contains working inference code, but we need to create compatible training code within the finetrainers framework as a new training type.

## Key Components

### Existing Components
1. **A2 Inference Implementation**: 
   - Located in the A2/ folder
   - Contains models/transformer_a2.py and models/pipeline_a2.py
   - Includes working inference scripts (infer.py, infer_MGPU.py)
   - Provides a Gradio demo (app.py)

2. **Finetrainers Framework**:
   - Already supports different training types for Wan model:
     - SFT training (standard fine-tuning)
     - Control training (augments latent noise tensors along control dimensions)
   - Provides a complete training infrastructure we can leverage

### Project Goals
- Create A2 training code as a new training type in the finetrainers framework
- Leverage the existing Wan model without creating a separate model type
- Extend the training framework to handle Elements-to-Video (E2V) data
- Ensure minimal changes while maintaining full compatibility with Wan

### Technical Requirements
1. A2 Training Approach Specifics:
   - **Multiple Reference Image Handling**: Process and condition on multiple images
     - Person/character reference image
     - Object reference image
     - Background reference image
   - **Dual Feature Processing Path**:
     - Semantic features (via CLIP) for global representation
     - Spatial features (via VAE) for local details and temporal conditioning
   - **Key Integration Points**:
     - Uses custom `WanAttnProcessor2_0` with support for image attention
     - Combines multiple references via concatenation in the channel dimension
     - Adds masking for frame conditioning similar to control training

2. Finetrainers Integration Points:
   - Use the existing Wan model specification (WanModelSpecification)
   - Build on control_trainer pattern which already handles channel concatenation
   - Add E2V specific dataset handling for multiple reference images
   - Extend configuration for A2-specific parameters (like number and type of references)
   - Create specialized preprocessors for handling CLIP and VAE encoding paths

3. Compatibility Considerations:
   - A2 and Wan share the identical transformer architecture
   - All differences are in data preprocessing and conditioning, not the architecture
   - Need to ensure consistency between training and inference pipelines
   - Special handling needed for combined condition latents

## Dataset Structure
The E2V (Elements-to-Video) training data uses a base identifier with different suffixes:

```
001.mp4                # Target video
001.txt                # Text prompt
001_mask.png           # Person/character reference
001_object.png         # Object reference
001_background.png     # Background/scene reference
```

Each identifier (e.g., "001") links together:
- A target video
- A text prompt
- Multiple reference images with various suffixes

The suffixes can vary across datasets (e.g., `_person.png` or `_mask.png` for character references), requiring our implementation to support flexible suffix mapping.

## Proposed Configuration Format
We'll extend the existing training.json format to support E2V training with minimal changes:

```json
{
  "datasets": [
    {
      "data_root": "path/to/e2v/dataset", 
      "dataset_type": "e2v",
      "target_resolution": [480, 704],
      "auto_frame_buckets": true,
      "clip_resolution": [512, 512],
      "reshape_mode": "bicubic",
      "remove_common_llm_caption_prefixes": true,
      "elements": [
        {
          "name": "main_subject", 
          "suffixes": ["_dog.png", "_person.png", "_mask.png"],
          "vae_repeat": 1,
          "position": 0,
          "clip_process": "center_crop"
        },
        {
          "name": "secondary",
          "suffixes": ["_object.png", "_toy.png"],
          "vae_repeat": 4, 
          "position": 1,
          "clip_process": "pad_white"
        },
        {
          "name": "environment",
          "suffixes": ["_background.png", "_scene.png"],
          "vae_repeat": 4,
          "position": 2,
          "clip_process": "letterbox" 
        }
      ],
      "vae_combine": "before",
      "visualization": {
        "enabled": true,
        "output_dir": "visualizations/{run_id}",
        "frequency": 100,
        "processors": [
          {
            "type": "latent_save",
            "data": ["input_latents", "predicted_latents"],
            "format": "pt"
          },
          {
            "type": "vae_decode",
            "data": ["input_latents", "predicted_latents"],
            "frames": [0, -1],
            "format": "png"
          }
        ]
      }
    }
  ]
}
```

### Visualization Configuration

The `visualization` section enables saving intermediate representations during training:

- **Integration Point**: Inside the `_train` method of `A2Trainer`, after forward and loss computation
- **Access Point**: Similar to checkpointing mechanism but for visualization data
- **Available Data**:
  - `input_latents`: Original input latents
  - `noisy_latents`: Latents with noise applied
  - `conditioned_latents`: Latents with conditioning
  - `predicted_latents`: Model predictions
  - `reference_images`: Original reference images
  
This approach leverages the training loop where all necessary data is already available, requiring minimal code changes. Each processor is defined by a `type` (the method to call) with all other fields passed as kwargs, keeping the implementation concise and extensible. 

The `frequency` parameter controls how often visualizations are generated (every N steps), minimizing performance impact while providing useful debugging information.
```

Key features of this configuration:

1. Uses the existing dataset structure with minimal extensions
2. Introduces a new `dataset_type: "e2v"` for Elements-to-Video datasets
3. Defines global target resolution with automatic frame bucket detection
4. Configures CLIP processing resolution globally
5. Specifies elements with flexible naming and dynamic suffix patterns
6. Provides element-specific configuration for:
   - VAE repetition counts
   - Positioning in the condition sequence
   - CLIP preprocessing method
7. Completely agnostic to what the elements actually represent (people, dogs, objects, etc.)
8. Global VAE combine mode setting

## Implementation Strategy
1. **A2 Trainer Development**:
   - Create new `a2_trainer` module alongside existing training types
   - Develop specialized dataset handling for multiple reference images
   - Implement processors for the E2V conditioning approach
   - Modify the forward pass to handle the dual-path (CLIP + VAE) conditioning
   - Leverage existing control training code for channel concatenation

2. **Data Processing**:
   - Create specialized dataset classes for multiple reference loading
   - Design flexible data loaders that can handle dynamic suffix patterns
   - Support configuration-driven element type mapping (e.g., mapping `_mask.png` or `_person.png` to "person")
   - Develop image preparation utilities for both semantic and spatial paths
   - Implement frame conditioning similar to control training pattern
   - Support variable numbers of reference images (1-3) as in A2 inference code

3. **Training Configuration**:
   - Add A2-specific training types to configuration
   - Create flexible configuration options for reference image types and naming patterns
   - Design training.json to support dynamic suffix mapping
   - Support various combination modes (as seen in inference code)
   - Allow specification of which element types are required vs. optional

4. **Testing and Validation**:
   - Develop specialized validation dataset for E2V tasks
   - Ensure compatibility with A2 inference code
   - Create test cases with various naming patterns to verify flexibility
   - Test handling of missing reference elements

## Data Processing Pipeline for A2 Training

### Reference Image Processing in A2

1. **VAE Spatial Pathway**:
   - Reference images are resized to match video dimensions (target_resolution)
   - Images are arranged in a sequence based on position parameter
   - Each image is repeated N times based on vae_repeat value
   - This creates a mini-video of reference images
   - Zero padding is added to match the target video's frame count
   - The entire sequence is encoded through VAE (like a video)
   - A frame mask is created to identify which frames are references
   - The mask and encoded mini-video are concatenated along the channel dimension
   - This combined tensor serves as conditioning for the model

2. **CLIP Semantic Pathway**:
   - Reference images are resized to clip_resolution
   - Each image is processed through CLIP vision encoder
   - The resulting embeddings provide semantic information
   - These embeddings feed into cross-attention layers

3. **Combined Effect**:
   - VAE path provides spatial details and structure
   - CLIP path provides high-level semantic understanding
   - Together they enable accurate preservation of reference elements

## Key Files to Create/Modify
- `finetrainers/trainer/a2_trainer/` - New training type implementation
- `finetrainers/trainer/a2_trainer/config.py` - Configuration for A2 training
- `finetrainers/trainer/a2_trainer/trainer.py` - A2 training loop
- `finetrainers/trainer/a2_trainer/data.py` - Data handling for multiple references
- `finetrainers/data/` - Add E2V dataset implementation
- `finetrainers/processors/clip.py` - Add/modify CLIP processor for image embeddings
- `finetrainers/config.py` - Add A2 training types
- `tests/trainer/test_a2_trainer.py` - Tests for the new implementation

## Implementation Analysis from Code Review
After reviewing the code, we've confirmed:

1. **Architectural Consistency**:
   - The A2Model class in `transformer_a2.py` extends Wan with identical parameters
   - Core transformer blocks and processing are unchanged
   - Primary differences are in data preparation and cross-attention handling

2. **Data Processing Insights**:
   - A2 concatenates multiple reference images with padding in VAE space
   - Uses CLIP vision encoder for semantic conditioning
   - Applies masking for specific frames similar to control training

3. **Integration Approach**:
   - We should follow the control_trainer pattern, which already has channel concatenation
   - The implementation should be a new training type rather than a new model
   - Code should be modular so most components can be reused with minimal changes

## Framework Reference Guide

### Core Framework Concepts

#### 1. Training Type Pattern
The framework organizes training approaches as distinct "types" rather than different models:

- **Reference**: `finetrainers/trainer/__init__.py` imports both `SFTTrainer` and `ControlTrainer` as different training types for the same models.

- **Concept**: Each training type implements a specific approach to fine-tuning but uses the same underlying model architecture. Our A2 training follows this pattern.

- **Application**: Create `a2_trainer` alongside existing types, not as a model variant.

#### 2. Model Specification Pattern
Models are defined through specification classes that handle model loading and conditioning:

- **Reference**: `finetrainers/models/wan/base_specification.py` defines the base Wan model, while `control_specification.py` extends it for control training.

- **Key Methods**: 
  - `load_diffusion_models()`: Loads transformer models with appropriate configurations
  - `forward()`: Defines how inputs flow through the model
  - `prepare_conditions()` and `prepare_latents()`: Handle preprocessing for different input types

- **Application**: Reuse the `WanControlModelSpecification` which already handles channel concatenation.

#### 3. Data Processing Pipeline
The framework separates data processing into distinct stages:

- **Reference**: `finetrainers/trainer/control_trainer/data.py` shows how control training handles conditioning data.

- **Key Components**:
  - Dataset classes return raw data
  - Preprocessors apply transformations 
  - Processors handle specific encodings (like `CannyProcessor`)

- **Application**: Create `IterableE2VDataset` and appropriate processors for reference images.

#### 4. Configuration System
Configurations cascade from general to specific:

- **Reference**: `finetrainers/trainer/control_trainer/config.py` defines training configuration classes.

- **Pattern**: 
  - Base configuration defines common parameters
  - Specialized configurations extend with specific needs

- **Application**: Ensure our dataset config extensions follow existing patterns.

### A2 Inference Code References

- **Reference Processing Function**: The `prepare_latents` function in `A2/models/pipeline_a2.py` (lines 288-394) handles all reference processing.

- **Mini-Video Creation**: At lines 330-347, references are arranged as a sequence and repeated based on the `vae_repeat` parameter. For example, with 3 reference images and `vae_repeat=True`, the first reference appears once, while the second and third are each repeated 4 times.

- **VAE Encoding**: At line 356, the entire reference sequence is encoded through VAE, treating it like a video: `latent_condition = retrieve_latents(self.vae.encode(video_condition), generator)`.

- **Frame Masking**: Lines 372-392 create a mask tensor where reference frames are marked with 1's and others with 0's. This mask is later used during conditioning.

- **Channel Concatenation**: At line 394, the mask and encoded latents are concatenated along the channel dimension: `return latents, torch.concat([mask_lat_size, latent_condition], dim=1)`.

- **CLIP Pathway**: Lines 570-575 show how CLIP embeddings from multiple references are concatenated and passed to the model via `encoder_hidden_states_image`.

- **Model Input**: Line 610 demonstrates how the latents and condition tensor are concatenated before being passed to the model: `latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)`.

## Development Guidelines

### Core Principles
- The WAN model IS the A2 model - no new model implementation is needed
- A2 is a training paradigm focused on multiple reference conditioning
- Maintain compatibility with existing inference code in A2/ directory

### Code Quality Standards
- **Minimal Code**: Use surgical precision - 10 lines is better than 100 
- **No Implicit Defaults**: Fail explicitly rather than using hidden default values
- **Framework Adherence**: Do not break existing framework patterns or functionality
- **Clean Separation**: Keep concerns separated and modules focused
- **No Code Pollution**: Avoid unnecessary verbosity and complexity
- **Error Handling**: Fail fast and explicitly when inputs are invalid
- **Testing**: Always run lint and typecheck commands to ensure code quality

### Implementation Approach
- Follow existing patterns from SFT and Control trainers
- Reuse code where appropriate, extend where necessary
- Minimize changes to core framework
- Prioritize readability and maintainability over cleverness