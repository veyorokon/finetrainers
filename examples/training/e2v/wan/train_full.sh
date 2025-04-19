#!/bin/bash

# Elements-to-Video (E2V) Full Fine-tuning Script Example for Wan Model

# ===== Model Configuration =====
MODEL_DIR="/dev/shm/models"
MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" 
PRETRAINED_MODEL_PATH="$MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers"

# ===== Training Configuration =====
TRAINING_TYPE="e2v-full-finetune"
OUTPUT_DIR="./output/e2v_wan_full"
CONFIG_FILE="./sample_config.json"
SEED=42

# ===== Optimization Configuration =====
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_TRAIN_STEPS=5000
LR=1e-5
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=100
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=4
CLIP_GRAD_NORM=1.0
WEIGHT_DECAY=1e-2

# ===== E2V Specific Configuration =====
E2V_TYPE="dual"
FRAME_CONDITIONING_TYPE="full"
FRAME_CONDITIONING_CONCATENATE_MASK=true

# ===== Checkpoint Configuration =====
CHECKPOINT_SAVE_STEPS=1000
CHECKPOINT_SAVE_TOTAL_LIMIT=3
CHECKPOINT_SAVE_WITH_OPTIMIZER=true

# ===== Validation Configuration =====
VALIDATION_STEPS=500
MAX_VALIDATION_BATCHES=3

# ===== Multi-GPU Configuration =====
# Required for full fine-tuning
export CUDA_VISIBLE_DEVICES=0,1,2,3
DISTRIBUTED_TYPE="fsdp"
NUM_PROCESSES=4
FSDP_OFFLOAD_PARAMS=false  # Set to true for CPU offloading if OOM issues occur

# --------------------------------------
# Run the training
# --------------------------------------
python ../../../train.py \
    --model_name="$MODEL_NAME" \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL_PATH" \
    --training_type="$TRAINING_TYPE" \
    --output_dir="$OUTPUT_DIR" \
    --dataset_configs="$CONFIG_FILE" \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    --max_train_steps="$MAX_TRAIN_STEPS" \
    --lr="$LR" \
    --lr_scheduler="$LR_SCHEDULER" \
    --lr_warmup_steps="$LR_WARMUP_STEPS" \
    --mixed_precision="$MIXED_PRECISION" \
    --dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    --clip_grad_norm="$CLIP_GRAD_NORM" \
    --weight_decay="$WEIGHT_DECAY" \
    --e2v_type="$E2V_TYPE" \
    --frame_conditioning_type="$FRAME_CONDITIONING_TYPE" \
    ${FRAME_CONDITIONING_CONCATENATE_MASK:+--frame_conditioning_concatenate_mask} \
    --checkpoint_save_steps="$CHECKPOINT_SAVE_STEPS" \
    --checkpoint_save_total_limit="$CHECKPOINT_SAVE_TOTAL_LIMIT" \
    ${CHECKPOINT_SAVE_WITH_OPTIMIZER:+--checkpoint_save_with_optimizer} \
    --validation_steps="$VALIDATION_STEPS" \
    --max_validation_batches="$MAX_VALIDATION_BATCHES" \
    --seed="$SEED" \
    --distributed_type="$DISTRIBUTED_TYPE" \
    --num_processes="$NUM_PROCESSES" \
    ${FSDP_OFFLOAD_PARAMS:+--fsdp_offload_params}