#!/bin/bash
#tmux new-session -d -s e2v_train 'bash /workspace/finetrainers/examples/training/e2v/wan/train_lora.sh > /workspace/e2v_log.txt 2>&1' 

#git fetch && git stash && git pull && tmux new-session -d -s e2v_train 'bash /workspace/finetrainers/examples/training/e2v/wan/train_lora.sh > /workspace/e2v_log.txt 2>&1'  

set -e -x

# Global project path - set this to your project root
PROJECT_PATH="/workspace/finetrainers"

# Environment variables
# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="INFO"

# Finetrainers supports multiple backends for distributed training
BACKEND="accelerate"

# In this setting, we're using 4 GPUs for DeepSpeed training
NUM_GPUS=1
CUDA_VISIBLE_DEVICES="0"

# Dataset configurations - use project-relative paths (relative to the project root)
TRAINING_DATASET_CONFIG="examples/training/e2v/wan/training.json"
VALIDATION_DATASET_FILE="examples/training/e2v/wan/validation.json" # Replace with actual validation file

# Model arguments
model_cmd=(
  --model_name "wan"
  --pretrained_model_name_or_path "/dev/shm/models/"
)

# E2V arguments
e2v_cmd=(
  --e2v_type "dual"
  --frame_conditioning_type "full"
  --frame_conditioning_concatenate_mask
)

# LoRA arguments
lora_cmd=(
  --rank 64
  --lora_alpha 64
  --target_modules "blocks.*(to_q|to_k|to_v|to_out.0)"
)

# Dataset arguments
dataset_cmd=(
  --dataset_config "$TRAINING_DATASET_CONFIG"
  --dataset_shuffle_buffer_size 32
  --cache_dir "/workspace/e2v/cache/"
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
  --training_type "e2v-lora"
  --seed 42
  --batch_size 1
  --train_steps 10000
  --gradient_accumulation_steps 4
  --gradient_checkpointing
  --checkpointing_steps 1000
  --checkpointing_limit 5
  # --resume_from_checkpoint 3000
)

# Optimizer arguments
optimizer_cmd=(
  --optimizer "adamw"
  --lr 5e-5
  --lr_scheduler "cosine"
  --lr_warmup_steps 500
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
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
  --tracker_name "finetrainers-e2v-wan-lora"
  --output_dir "/workspace/e2v_wan_lora"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
  --mixed_precision "bf16"
)

# DeepSpeed configuration
deepspeed_cmd=(
  --deepspeed_config "accelerate_configs/deepspeed.yaml"
)

# Set CUDA devices and execute with torchrun
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_backend c10d \
  --rdzv_endpoint="localhost:29500" \
    train.py \
    --parallel_backend "$BACKEND" \
    "${model_cmd[@]}" \
    "${e2v_cmd[@]}" \
    "${lora_cmd[@]}" \
    "${dataset_cmd[@]}" \
    "${dataloader_cmd[@]}" \
    "${diffusion_cmd[@]}" \
    "${training_cmd[@]}" \
    "${optimizer_cmd[@]}" \
    "${validation_cmd[@]}" \
    "${miscellaneous_cmd[@]}" \
    "${deepspeed_cmd[@]}"

echo -ne "-------------------- Finished executing script --------------------\n\n"