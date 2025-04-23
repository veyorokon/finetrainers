# E2V Trainer Implementation Guide

  ## Project Goals

  We're implementing Elements-to-Video (E2V) training within the finetrainers framework. This approach extends the
  existing ControlTrainer with the additional capability to condition on multiple reference images using both VAE
  (spatial) and CLIP (semantic) pathways.

  The core principle is to **leverage inheritance and the existing control trainer infrastructure** while adding only
  the minimal necessary code for E2V-specific functionality.

  ## Architecture Approach

  ### Inheritance-Based Design

  The E2V trainer is implemented as a direct extension of the ControlTrainer:

  ```python
  class E2VTrainer(ControlTrainer):
      """Elements-to-Video trainer that extends ControlTrainer with E2V-specific functionality."""

  This allows us to:
  1. Reuse all of ControlTrainer's existing functionality (model loading, data processing, training loop)
  2. Override only the specific methods needed for E2V functionality
  3. Maintain full compatibility with the framework's patterns

  Key Components to Implement

  1. E2VTrainer: Extends ControlTrainer with CLIP processing capability
  2. IterableE2VDataset: Wraps dataset with configuration-driven element identification
  3. Model Specification Extensions: Adds CLIP model loading functionality

  Configuration-Driven Approach

  E2V training is configured through a JSON file that specifies:
  1. Dataset elements (object, background, captions)
  2. Conditioning approaches (frame conditioning, CLIP, text)
  3. Element-specific processing parameters

  Implementation Plan

  Files to Create/Modify

  1. finetrainers/trainer/e2v_trainer/trainer.py
    - Extends ControlTrainer
    - Overrides only necessary methods
    - Adds CLIP coordination
  2. finetrainers/trainer/e2v_trainer/data.py
    - Implements IterableE2VDataset
    - Handles configuration-driven element identification
    - Preprocesses elements based on conditioning type
  3. finetrainers/trainer/e2v_trainer/config.py
    - Defines E2VConfig extending ControlConfig
    - Adds E2V-specific configuration parameters
  4. finetrainers/models/wan/e2v_specification.py
    - Extends WanControlModelSpecification
    - Adds CLIP model loading methods

  Minimal Method Overrides

  Only override methods that need E2V-specific functionality:

  1. _prepare_models: Add CLIP model loading
  2. _prepare_dataset: Use E2V dataset wrapper
  3. _prepare_data: Add CLIP processing phase
  4. _forward_pass: Integrate CLIP embeddings

  Implementation Principles

  1. Always Reference Control Trainer

  When implementing any E2V-specific functionality, always refer to how ControlTrainer handles similar tasks:

  # Example: Memory management
  def _delete_components(self, component_names=None):
      # Check how control trainer does it
      return super()._delete_components(component_names)

  2. Minimal Code Additions

  Add only the code necessary for E2V-specific functionality:
  - CLIP model loading and processing
  - Configuration-driven element handling
  - Multiple reference processing

  3. Maintain Framework Compatibility

  Follow established framework patterns:
  - Use same model device movement approach
  - Match memory management patterns
  - Keep consistent with parallel processing support

  Debugging Approach

  When encountering issues:

  1. Check Control Trainer First: See how the control trainer handles the problematic area
  2. Adapt Solution: Adapt the control trainer's solution to E2V context
  3. Maintain Patterns: Keep consistent with framework patterns and conventions

  For example, if facing a model coordination issue:
  Control Trainer Solution: Uses _move_components_to_device() and _delete_components()
  E2V Adaptation: Use the same pattern but add CLIP model to the component list

  Configuration Format

  {
    "datasets": [
      {
        "data_root": "/workspace/dataset",
        "dataset_type": "video_references",

        "elements": [
          {
            "name": "object",
            "suffixes": ["_object.png"],
            "required": true,
            "conditioning": "reference",
            "vae": { "repeat": 4, "position": 0 },
            "clip": { "position": 0 }
          },
          {
            "name": "background",
            "suffixes": ["_background.png"],
            "required": false,
            "conditioning": "reference",
            "vae": { "repeat": 1, "position": 1 }
          },
          {
            "name": "captions",
            "suffixes": [".txt"],
            "required": true,
            "conditioning": "text"
          }
        ],

        "conditioning": {
          "reference": {
            "type": "frame",
            "frame_conditioning_type": "full",
            "frame_conditioning_concatenate_mask": true,
            "resolution": [480, 854]
          },
          "text": {
            "type": "text",
            "remove_common_llm_caption_prefixes": true
          },
          "clip": {
            "type": "clip",
            "resolution": [224, 224],
            "preprocessor": "center_crop"
          }
        }
      }
    ]
  }

  Extensibility

  This design can be easily extended to support new conditioning types by:

  1. Adding new conditioning types in the configuration
  2. Implementing the corresponding preprocessing in the dataset wrapper
  3. Adding model loading and processing in the trainer

  No core architectural changes would be needed to add new conditioning types.

  Framework Compatibility

  The implementation maintains compatibility with:
  - Accelerate framework
  - Checkpointing mechanism
  - Distributed training
  - Memory management utilities

  Summary

  The E2V trainer implementation focuses on leveraging inheritance from ControlTrainer to minimize code while adding
  specific functionality for E2V training. By following this guide, you'll maintain framework compatibility while
  creating a flexible, configuration-driven trainer for Elements-to-Video tasks.

  This CLAUDE.md file provides a comprehensive guide to implementing the E2V trainer with minimal code by leveraging
  the control trainer as much as possible, while maintaining clear principles for debugging and extension.