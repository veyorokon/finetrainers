#!/bin/bash

set -e -x

# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="INFO"

# Finetrainers supports multiple backends for distributed training
BACKEND="fsdp"

# In this setting, we're using 4 GPUs for FSDP training
NUM_GPUS=4
CUDA_VISIBLE_DEVICES="0,1,2,3"

# Check the JSON files for the expected JSON format
TRAINING_DATASET_CONFIG="./sample_config.json"
VALIDATION_DATASET_FILE="./sample_config.json" # Replace with actual validation file

# Model arguments
model_cmd=(
  --model_name "wan"
  --pretrained_model_name_or_path "/dev/shm/models/Wan2.1-T2V-1.3B-Diffusers"
)

# E2V arguments
e2v_cmd=(
  --e2v_type "dual"
  --frame_conditioning_type "full"
  --frame_conditioning_concatenate_mask
)

# Dataset arguments
dataset_cmd=(
  --dataset_config "$TRAINING_DATASET_CONFIG"
  --dataset_shuffle_buffer_size 32
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 4
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

# Training arguments
training_cmd=(
  --training_type "e2v-full-finetune"
  --seed 42
  --batch_size 1
  --train_steps 5000
  --gradient_accumulation_steps 4
  --gradient_checkpointing
  --checkpointing_steps 1000
  --checkpointing_limit 3
  --checkpointing_with_optimizer
  # --resume_from_checkpoint 3000
  --enable_slicing
  --enable_tiling
)

# Optimizer arguments
optimizer_cmd=(
  --optimizer "adamw"
  --lr 1e-5
  --lr_scheduler "cosine"
  --lr_warmup_steps 100
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-2
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# Validation arguments
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 500
  --max_validation_batches 3
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --tracker_name "finetrainers-e2v-wan-full"
  --output_dir "./output/e2v_wan_full"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
  --mixed_precision "bf16"
)

# FSDP configuration
fsdp_cmd=(
  # --fsdp_offload_params  # Uncomment for CPU offloading if OOM issues occur
)

# Set CUDA devices and execute with torchrun
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:29500" \
  ../../../train.py \
    --parallel_backend "$BACKEND" \
    "${model_cmd[@]}" \
    "${e2v_cmd[@]}" \
    "${dataset_cmd[@]}" \
    "${dataloader_cmd[@]}" \
    "${diffusion_cmd[@]}" \
    "${training_cmd[@]}" \
    "${optimizer_cmd[@]}" \
    "${validation_cmd[@]}" \
    "${miscellaneous_cmd[@]}" \
    "${fsdp_cmd[@]}"

echo -ne "-------------------- Finished executing script --------------------\n\n"