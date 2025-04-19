#!/bin/bash

# Elements-to-Video (E2V) Training Script Example for Wan Model

# Configuration
MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUTPUT_DIR="./output/e2v_wan_lora"
CONFIG_FILE="./sample_config.json"
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_TRAIN_STEPS=10000
LR=5e-5
RANK=64

# Set up environment if needed
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the training
python ../../../train.py \
    --model_name="$MODEL_NAME" \
    --training_type="e2v-lora" \
    --output_dir="$OUTPUT_DIR" \
    --lr="$LR" \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    --max_train_steps="$MAX_TRAIN_STEPS" \
    --validation_steps=500 \
    --checkpoint_save_steps=1000 \
    --dataset_configs="$CONFIG_FILE" \
    --e2v_type="dual" \
    --frame_conditioning_type="full" \
    --frame_conditioning_concatenate_mask \
    --mixed_precision="bf16" \
    --rank="$RANK" \
    --lora_alpha="$RANK" \
    --dataloader_num_workers=4 \
    --seed=42