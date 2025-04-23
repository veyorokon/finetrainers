#!/bin/bash

# Test script for E2V trainer
# This script uses minimal parameters to test if the E2V trainer can initialize correctly

# Set up environment
BASEDIR=$(dirname "$0")
cd "$BASEDIR/../../../.."
ROOT_DIR=$(pwd)

echo "Testing E2V trainer from: $ROOT_DIR"

# Run trainer in test mode
python train.py \
    --model_name="wan" \
    --training_type="e2v-lora" \
    --rank=4 \
    --lora_alpha=8 \
    --target_modules="transformer_blocks.*attn.(q|k|v|out)" \
    --frame_conditioning_type="full" \
    --frame_conditioning_concatenate_mask \
    --output_dir="outputs/e2v_test" \
    --dataset_config="examples/training/e2v/wan/test_training.json" \
    --scheduler="euler" \
    --train_steps=1 \
    --gradient_accumulation_steps=1 \
    --lr=1e-5 \
    --checkpointing_steps=0 \
    --seed=42 \
    --verbose \
    --debug_mode