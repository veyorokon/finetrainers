# E2V Trainer

The Elements-to-Video (E2V) Trainer is a specialized training type that enables training video diffusion models using multiple reference images (elements) as conditioning inputs. It follows the approach demonstrated in A2, which is a specialized training approach for the Wan model.

## Overview

E2V training enables a model to generate videos from multiple reference images, such as:
- Main subject/character reference image
- Object reference image
- Background/scene reference image

The approach leverages two conditioning pathways:
1. **CLIP Pathway**: Processes images through a CLIP vision encoder to extract semantic features
2. **VAE Pathway**: Processes images through VAE encoding to preserve spatial details

## Key Concepts

### Elements
In E2V training, "elements" are the reference images used to condition video generation. Each element has specific properties:
- **Name**: Identifier for the element (e.g., "main_subject", "object", "background")
- **Suffixes**: File suffixes to identify this element type (e.g., ["_person.png", "_mask.png"])
- **Required**: Whether this element is mandatory
- **VAE Configuration**: How this element should be processed in the VAE pathway (position, repeat count)
- **CLIP Configuration**: How this element should be processed in the CLIP pathway (preprocessing type)

### Conditioning Pathways

#### VAE Pathway
- Encodes images to preserve spatial details and structure
- Creates a "mini-video" of reference frames by:
  - Arranging elements in a sequence based on position
  - Applying repetition to create multiple frames for each element
  - Encoding the sequence through VAE
  - Creating a frame mask for identifying reference frames
  - Concatenating the mask and encoded latents

#### CLIP Pathway
- Processes images through CLIP vision encoder to extract semantic features
- Features are concatenated and passed as key/values to cross-attention layers

## Configuration Format

E2V training uses a JSON configuration format:

```json
{
  "datasets": [
    {
      "data_root": "path/to/e2v/dataset", 
      "dataset_type": "e2v",
      "target_resolution": [480, 854],
      "auto_frame_buckets": true,
      "reshape_mode": "bicubic",
      
      "elements": [
        {
          "name": "main_subject", 
          "suffixes": ["_dog.png", "_person.png", "_mask.png"],
          "required": true,
          "vae": {
            "repeat": 4, 
            "position": 0
          },
          "clip": {
            "preprocess": "center_crop"
          }
        },
        {
          "name": "secondary",
          "suffixes": ["_object.png", "_toy.png"],
          "required": false,
          "vae": {
            "repeat": 4, 
            "position": 1
          },
          "clip": {
            "preprocess": "pad_white"
          }
        },
        {
          "name": "environment",
          "suffixes": ["_background.png", "_scene.png"],
          "required": false,
          "vae": {
            "repeat": 4, 
            "position": 2
          },
          "clip": {
            "preprocess": "letterbox"
          }
        }
      ],
      
      "processors": {
        "vae": {
          "resolution": [480, 854],
          "combine": "before",
          "default_preprocess": "resize",
          "frame_conditioning": "full",
          "frame_index": 0,
          "concatenate_mask": true
        },
        "clip": {
          "resolution": [512, 512],
          "default_preprocess": "center_crop"
        }
      }
    }
  ]
}
```

## Dataset Structure

The E2V training dataset should contain:
- Target videos
- Text prompts
- Reference images with specific suffixes

Example file structure:
```
dataset/
  ├── 001.mp4                # Target video
  ├── 001.txt                # Text prompt
  ├── 001_mask.png           # Person/character reference
  ├── 001_object.png         # Object reference
  ├── 001_background.png     # Background/scene reference
  ├── 002.mp4
  ├── 002.txt
  ├── 002_mask.png
  ├── ...
```

## Usage

### Command Line Arguments

In addition to standard finetrainers arguments, E2V training supports:

- `--e2v_type`: Type of E2V processing ("vae", "clip", or "dual")
- `--frame_conditioning_type`: Type of frame conditioning ("index", "prefix", "random", "first_and_last", "full")
- `--frame_conditioning_index`: Index for frame conditioning (when using "index" type)
- `--frame_conditioning_concatenate_mask`: Whether to concatenate frame mask with latents

### Example Command

```bash
python train.py \
    --model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --training_type="e2v-lora" \
    --output_dir="./output/e2v_wan_lora" \
    --dataset_configs="./config.json" \
    --e2v_type="dual" \
    --frame_conditioning_type="full" \
    --frame_conditioning_concatenate_mask \
    --mixed_precision="bf16" \
    --rank=64 \
    --lora_alpha=64
```

## Implementation Details

The E2V trainer builds on the same model architecture as control training, but with specialized data processing:

1. The trainer uses `WanControlModelSpecification` (same as control training)
2. The transformer patch embedding is expanded to handle concatenated control channels
3. The implementation uses the same frame conditioning approaches as control training
4. The CLIP pathway leverages the model's existing image-to-video capability