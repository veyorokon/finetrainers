# E2V Trainer Implementation Guide

## Project Goals

We're implementing Elements-to-Video (E2V) training within the finetrainers framework. This approach extends the existing ControlTrainer with the additional capability to condition on multiple reference images using both VAE (spatial) and CLIP (semantic) pathways.

The core principle is to **leverage inheritance and the existing control trainer infrastructure** while adding only the minimal necessary code for E2V-specific functionality.

## Architecture Approach

### Inheritance-Based Design

The E2V trainer is implemented as a direct extension of the ControlTrainer:

```python
class E2VTrainer(ControlTrainer):
    """Elements-to-Video trainer that extends ControlTrainer with E2V-specific functionality."""
```

This allows us to:
1. Reuse all of ControlTrainer's existing functionality (model loading, data processing, training loop)
2. Override only the specific methods needed for E2V functionality
3. Maintain full compatibility with the framework's patterns

### Key Components

1. **E2VTrainer**: Extends ControlTrainer with CLIP processing capability
   - Adds image encoder loading and processing
   - Implements optimized model coordination pattern
   - Handles E2V-specific configuration

2. **IterableE2VDataset**: Configuration-driven dataset wrapper
   - Identifies elements via configured suffixes
   - Preprocesses based on conditioning type
   - No model inference in preprocessing

3. **Configuration System**: Flexible, extensible configuration
   - Elements define dataset components to process
   - Conditioning types determine processing approach
   - Element-specific processor overrides

## Implementation Details

### Model Coordination Pattern

The E2V trainer uses an optimized model coordination pattern:

1. **Sequential Model Loading**: One model at a time on GPU
   - Text encoder → CLIP → VAE → transformer
   - Process all data for one model before moving to next
   - Explicit memory management between models

2. **Batch Processing**: Process similar data together
   - Group similar images/videos for efficient batching
   - Process all samples needing the same model at once

3. **Memory Optimization**: Careful resource management
   - Move models to CPU when not in use
   - Explicit memory freeing between stages
   - Reuse control trainer memory management utilities

### Configuration-Driven Element Processing

The trainer uses a flexible configuration system:

```json
{
  "elements": [
    {
      "name": "object",
      "suffixes": ["_object.png"],
      "required": true,
      "conditioning": "reference",
      "vae": { "repeat": 4, "position": 0 },
      "clip": { "position": 0 }
    }
  ],
  "conditioning": {
    "reference": {
      "type": "frame",
      "frame_conditioning_type": "full",
      "frame_conditioning_concatenate_mask": true
    }
  }
}
```

This configuration:
1. Identifies elements by filenames
2. Determines which processing to apply
3. Sets element-specific parameters
4. Defines global conditioning settings

## Files and Implementation

### 1. finetrainers/trainer/e2v_trainer/trainer.py

This file extends ControlTrainer with E2V-specific functionality:
- Overrides `_prepare_models()` to load CLIP model
- Implements `_prepare_data()` with optimized model coordination
- Adds helper methods for processing text, CLIP, and VAE data
- Handles E2V-specific model management and memory optimization

Key method: `_prepare_data()`
```python
def _prepare_data(self, preprocessor, data_iterator):
    # 1. Collect samples into buffer
    collected_samples = []
    
    # 2. Process all text data at once
    self._move_components_to_device([self.text_encoder])
    collected_samples = self._process_text_batch(collected_samples)
    self._move_components_to_device([self.text_encoder], "cpu")
    
    # 3. Process all CLIP data at once
    self._move_components_to_device([self.clip_model])
    collected_samples = self._process_clip_batch(collected_samples)
    self._move_components_to_device([self.clip_model], "cpu")
    
    # 4. Process all VAE data at once
    self._move_components_to_device([self.vae])
    collected_samples = self._process_vae_batch(collected_samples)
    self._move_components_to_device([self.vae], "cpu")
    
    # 5. Return to transformer for forward pass
    self._move_components_to_device([self.transformer])
    
    # Create iterators for training loop
    return iter(collected_samples), iter(collected_samples)
```

### 2. finetrainers/trainer/e2v_trainer/data.py

This file implements the dataset wrapper with configuration-driven element processing:
- Uses element suffixes to identify files
- Applies preprocessing based on conditioning type
- Handles text, image, and video preprocessing
- Returns preprocessed tensors ready for model inference

Key method: `_preprocess_elements()`
```python
def _preprocess_elements(self, data, element_files):
    processed = {}
    
    for element_name, file_info in element_files.items():
        element_config = file_info["config"]
        conditioning_type = element_config.get("conditioning")
        
        # Process element based on conditioning type
        if conditioning_type == "frame":
            self._process_frame_element(processed, element_name, file_info)
        elif conditioning_type == "clip":
            self._process_clip_element(processed, element_name, file_info)
        elif conditioning_type == "text":
            self._process_text_element(processed, element_name, file_info)
    
    return processed
```

### 3. finetrainers/trainer/e2v_trainer/config.py

This file defines the configuration classes for E2V training:
- Extends configuration from ControlTrainer
- Adds E2V-specific parameters
- Provides configuration handling for both LoRA and full fine-tuning

```python
class E2VConfig(ConfigMixin):
    """Configuration for E2V training, extending control configuration."""
    
    frame_conditioning_type: str = FrameConditioningType.FULL
    frame_conditioning_index: int = 0
    frame_conditioning_concatenate_mask: bool = True
    
    # E2V specific configuration
    elements_config: Dict = {}  # Will be populated from JSON
    conditioning_config: Dict = {}  # Will be populated from JSON
```

## Debugging Guide

When encountering issues, always refer to how ControlTrainer handles similar functionality:

### Memory Issues

If encountering OOM errors:
1. Check `_prepare_data()` in ControlTrainer
2. Make sure models are properly moved to CPU after use:
   ```python
   self._move_components_to_device([self.image_encoder], "cpu")
   utils.free_memory()
   ```
3. Verify sequential model loading pattern is preserved

### Data Processing Issues

If encountering data processing problems:
1. Check preprocessing in IterableControlDataset
2. Compare with IterableE2VDataset implementation
3. Verify element identification and preprocessing
4. Check how control conditions are prepared

### CLIP Integration Issues

If CLIP pathway isn't working:
1. Check model loading in `_prepare_models`
2. Verify CLIP model is correctly registered in model specification
3. Check `_process_clip_batch` implementation
4. Make sure CLIP features are properly combined

## Extension Guide

To add new conditioning types:

1. Add new element type in configuration:
   ```json
   {
     "name": "depth_map",
     "suffixes": ["_depth.png"],
     "conditioning": "depth"
   }
   ```

2. Add new conditioning type:
   ```json
   "conditioning": {
     "depth": {
       "type": "channel",
       "resolution": [480, 854],
       "preprocessor": "resize"
     }
   }
   ```

3. Add processing in `_process_<type>_element` method in the dataset

4. Handle new conditioning type in the trainer's `_process_<type>_batch` method

## Troubleshooting

When debugging, you should:

1. Check if the ControlTrainer has solved a similar problem
2. Look for framework utilities that handle the issue
3. Consider memory management implications
4. Check configuration structure and validation

Always remember that the E2V trainer is an extension of ControlTrainer, so most solutions can be adapted from the parent class implementation.

## Final Note

This implementation follows the principle of minimal modification while maintaining full compatibility with the framework. By extending ControlTrainer, we inherit all its robust functionality while adding only the necessary E2V-specific features. focus on fixing the actual issue rather than masking it with defaults that weren't specified by the user. 

When providing a commit message note the format: a header and body.
e.g.:  fix: add debugging to E2VLowRankConfig for target_modules parsing

Add minimal logging in map_args method to diagnose why target_modules argument isn't being parsed correctly when 
passed as a regex pattern string, while working properly in the ControlTrainer
Standard headers: [fix, logging, feat, doc, refactor]

You will be in one of those 5 modes. Before  you begin - announce which mode.

Fix: logical changes and logging
Logging: No logical changes to code just logging statements.
Feat: code changes to implement new feature
Doc: Documentation creating
Refactor: Clean, remove, restructure and simplify code


Mode requirements:
- For logging: no new if / else statements OR loops are allowed . if youre unsure about the fields - print the dir()