# Modular LoRA Strategy for E2V Model Customization

## Overview

This document outlines a modular approach to customizing the Elements-to-Video (E2V) model using LoRA (Low-Rank Adaptation) for product-focused video generation. The strategy leverages the dual-pathway architecture of the model to create specialized, non-overlapping adapters that can be combined flexibly at inference time.

## Architecture Background

The E2V model processes information through two primary pathways:

1. **Spatial Pathway**: Processes structural and spatial information via VAE encoding
   - Entry point: `patch_embedding` layer
   - Creates condition tensors that are concatenated with video latents
   - Handles detailed structure, movement, and composition

2. **Semantic Pathway**: Processes high-level semantic understanding via CLIP encoding
   - Entry points: `add_k_proj` and `add_v_proj` layers
   - Integrated via cross-attention mechanism
   - Handles style, aesthetic qualities, and visual characteristics

This dual-pathway architecture enables a targeted approach to LoRA training, allowing different aspects of generation to be modified independently.

## Adaptable Base + Modular Style Strategy

Our recommended approach consists of:

1. **One Adaptable Base LoRA** that works across multiple scenarios
2. **Multiple Style LoRAs** that can be swapped or combined

### Adaptable Base LoRA

**Target Modules**:
```
".*blocks\..*\.(to_q|to_k|to_v|to_out\.0|ffn\.)|.*patch_embedding.*"
```

**Purpose**:
- Handles composition, movement, and scene arrangement
- Works across multiple scenarios (product-only, multi-product, product+person)
- Serves as the foundation for all generations

**Training Data**:
- Balanced dataset containing diverse scenarios:
  - Single product videos (rotating, moving, demonstrating)
  - Multiple product interactions
  - Product with person interactions
  - Varied camera movements (pans, zooms, rotations)

**Technical Parameters**:
- Higher rank (32-64) to capture complex compositional elements
- Medium to larger dataset size (500-1000 videos)
- Focus on visual coherence and natural movement

### Modular Style LoRAs

**Target Modules**:
```
".*blocks\..*\.(add_k_proj|add_v_proj|norm_added_k).*"
```

**Purpose**:
- Modify visual style independently of composition
- Create a library of interchangeable aesthetic options
- Enable quick adaptation to different client preferences

**Example Style Variations**:

| Style Name | Characteristics | Training Focus |
|------------|-----------------|----------------|
| Cinematic-Dramatic | High contrast, selective lighting, shallow DoF | Film-like sequences with dramatic atmosphere |
| Bright-Commercial | High-key lighting, vibrant colors, clear visibility | Traditional commercial aesthetics |
| Moody-Atmospheric | Low-key lighting, atmospheric elements, mood emphasis | Emotional, evocative presentations |
| Technical-Precise | Neutral lighting, accurate colors, detail emphasis | Product functionality and features |
| Minimalist | Clean backgrounds, simple lighting, focus on form | Modern, sleek presentation style |
| Testimonial | Simple personable relatable | Unproduced DIY like style |


**Technical Parameters**:
- Lower rank (8-16) for faster training and efficient storage
- Smaller datasets (100-300 videos) with consistent aesthetic 
- Focus on visual style rather than movement or composition

## Implementation Guide

### Training the Adaptable Base

```bash
python train.py \
  --config=examples/training/e2v/product_base/training.json \
  --training_type=e2v \
  --use_peft=True \
  --rank=32 \
  --lora_alpha=32 \
  --target_modules=".*blocks\..*\.(to_q|to_k|to_v|to_out\.0|ffn\.)|.*patch_embedding.*"
```

**Dataset Considerations**:
- Include diverse product categories (electronics, fashion, cosmetics, etc.)
- Vary camera angles and movements
- Include both single product and multi-product scenarios
- Mix scenarios with and without people
- Ensure consistent high quality across all samples

### Training Style LoRAs

```bash
python train.py \
  --config=examples/training/e2v/styles/cinematic/training.json \
  --training_type=e2v \
  --use_peft=True \
  --rank=16 \
  --lora_alpha=16 \
  --target_modules=".*blocks\..*\.(add_k_proj|add_v_proj|norm_added_k).*"
```

**Dataset Considerations**:
- Focus on visual consistency within each style category
- Curate videos with strong examples of the target aesthetic
- Consistent lighting and color grading 
- Can be smaller datasets (style requires less variation than composition)

### Inference with Combined LoRAs

```python
from diffusers import A2Pipeline
import torch

# Initialize pipeline
pipe = A2Pipeline.from_pretrained(
    "skyreel/a2-base-model",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Load adaptable base LoRA
pipe.load_lora_weights("path/to/adaptable_base_lora", adapter_name="base")

# Load style LoRA
pipe.load_lora_weights("path/to/cinematic_style_lora", adapter_name="style")

# Optional: adjust influence of each adapter
pipe.set_adapters(["base", "style"], adapter_weights=[1.0, 0.7])

# Generate with combined adapters
output = pipe(
    prompt="Premium headphones floating in space with dynamic lighting",
    image_vae=[headphones_image, space_background],
    negative_prompt="low quality, blurry, distorted",
    num_inference_steps=30,
    guidance_scale=7.0,
)
```

### Mixing Multiple Style LoRAs

For even more customization, multiple style LoRAs can be combined:

```python
# Load multiple style LoRAs
pipe.load_lora_weights("path/to/cinematic_lora", adapter_name="cinematic")
pipe.load_lora_weights("path/to/vibrant_lora", adapter_name="vibrant")

# Blend styles with different weights
pipe.set_adapters(
    ["base", "cinematic", "vibrant"], 
    adapter_weights=[1.0, 0.6, 0.3]
)

# Create a hybrid style output
output = pipe(
    prompt="Smartwatch with futuristic interface, dynamic lighting",
    image_vae=[watch_image, tech_background],
    negative_prompt="low quality, blurry",
    num_inference_steps=30,
)
```

## Scenario Control via Prompting

With a single adaptable base, specific scenarios can be controlled through prompt engineering:

| Scenario | Example Prompt |
|----------|----------------|
| Single Product Focus | "Luxury watch rotating on marble surface, dramatic spotlight" |
| Multi-Product | "Smartphone and wireless earbuds floating in synchronized movement" |
| Product with Person | "Person demonstrating smart fitness device, interactive display" |

The adaptable base LoRA will handle these different compositions while maintaining quality across scenarios, and the style LoRA adds the desired aesthetic treatment.

## Advantages of This Approach

1. **Simplicity in Management**: One base LoRA instead of multiple specialized versions
2. **Flexible Style Application**: Swap styles instantly without retraining
3. **Non-Interference**: Style changes don't affect composition quality
4. **Efficiency**: Style LoRAs are small and train quickly on limited data
5. **Expandability**: New styles can be added without affecting existing capabilities
6. **Fine Control**: Adjust influence of base vs. style through adapter weights

## Best Practices

1. **Base LoRA Training**:
   - Use diverse, high-quality product videos
   - Include all scenarios you want to support
   - Focus on natural movement and composition
   - Higher rank for more capacity (32-64)

2. **Style LoRA Training**:
   - Use videos with consistent aesthetic qualities
   - Strong examples of target visual style
   - Can use shorter clips focused on the look rather than narrative
   - Lower rank for efficiency (8-16)

3. **Prompt Engineering**:
   - Include clear references to desired composition
   - Add style-specific keywords that align with LoRA training
   - Use negative prompts to avoid unwanted elements

4. **Testing Process**:
   - Validate base LoRA across all scenario types
   - Test style LoRAs both individually and in combinations
   - Experiment with different adapter weights
   - Create a test set of standard scenarios to ensure consistency

## Technical Considerations

1. **Module Targeting Precision**:
   - Verify module names match your specific model version
   - Test non-overlapping target patterns to ensure clean separation

2. **Training Parameters**:
   - Base LoRA: higher learning rate at start, then decay
   - Style LoRAs: lower, consistent learning rate
   - Monitor validation loss to prevent overfitting

3. **Inference Optimization**:
   - Cache loaded LoRAs for faster switching
   - Pre-compute common style combinations

4. **Potential Issues**:
   - Style bleeding (style affecting composition): Reduce style adapter weight
   - Inconsistent quality across scenarios: Rebalance training data
   - Style not strong enough: Increase style adapter weight or retrain with more focused data

---

This modular approach provides maximum flexibility while maintaining efficiency in training and deployment. By separating composition from style, it enables rapid adaptation to different client needs while ensuring consistent quality across all generations.

Document Version: 1.0  
Last Updated: April 21, 2025