#!/bin/bash

# Elements-to-Video (E2V) DeepSpeed Training Script Example for Wan Model

# ===== Model Configuration =====
MODEL_DIR="/dev/shm/models"
MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" 
PRETRAINED_MODEL_PATH="$MODEL_DIR/Wan2.1-T2V-1.3B-Diffusers"

# ===== Training Configuration =====
TRAINING_TYPE="e2v-lora"  # Change to e2v-full-finetune for full fine-tuning
OUTPUT_DIR="./output/e2v_wan_deepspeed"
CONFIG_FILE="./sample_config.json"
SEED=42

# ===== Optimization Configuration =====
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
MAX_TRAIN_STEPS=10000
LR=5e-5
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=500
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=4

# ===== LoRA Configuration =====
RANK=64
LORA_ALPHA=$RANK
TARGET_MODULES="(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"

# ===== E2V Specific Configuration =====
E2V_TYPE="dual"
FRAME_CONDITIONING_TYPE="full"
FRAME_CONDITIONING_CONCATENATE_MASK=true

# ===== Checkpoint Configuration =====
CHECKPOINT_SAVE_STEPS=1000
CHECKPOINT_SAVE_TOTAL_LIMIT=5

# ===== Validation Configuration =====
VALIDATION_STEPS=500
MAX_VALIDATION_BATCHES=3

# ===== DeepSpeed Configuration =====
export CUDA_VISIBLE_DEVICES=0,1,2,3
DISTRIBUTED_TYPE="deepspeed"
NUM_PROCESSES=4
DEEPSPEED_CONFIG="../../../accelerate_configs/deepspeed.yaml"

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
    --rank="$RANK" \
    --lora_alpha="$LORA_ALPHA" \
    --target_modules="$TARGET_MODULES" \
    --e2v_type="$E2V_TYPE" \
    --frame_conditioning_type="$FRAME_CONDITIONING_TYPE" \
    ${FRAME_CONDITIONING_CONCATENATE_MASK:+--frame_conditioning_concatenate_mask} \
    --checkpoint_save_steps="$CHECKPOINT_SAVE_STEPS" \
    --checkpoint_save_total_limit="$CHECKPOINT_SAVE_TOTAL_LIMIT" \
    --validation_steps="$VALIDATION_STEPS" \
    --max_validation_batches="$MAX_VALIDATION_BATCHES" \
    --seed="$SEED" \
    --distributed_type="$DISTRIBUTED_TYPE" \
    --num_processes="$NUM_PROCESSES" \
    --deepspeed_config="$DEEPSPEED_CONFIG"