#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR="/root/LLaVA-RLHF/data_dir"
export MODEL_DIR="/root/LLaVA-RLHF/model_dir"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=2
export OMP_NUM_THREADS=2

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14-336
LM_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model/

# DATA CONFIG
PREFERENCE_DATA=vla_224_nrmse.json

# SAVE CONFIG
MODEL_NAME=LLaVA-RM-7b-224-full-finetune-fsdp

# WANDB CONFIG
export WANDB_PROJECT="llava-224-nrmse-ground"
export WANDB_NAME="$MODEL_NAME-$(date +%Y%m%d_%H%M%S)"
export WANDB_ENTITY="skyrobo"  # Replace with your wandb username or organization

# TRAINING CONFIG
NUM_EPOCHS=20
LEARNING_RATE=2e-5
BATCH_SIZE=4  # Increased for FSDP
GRAD_ACCUMULATION=1

# FSDP CONFIG
FSDP_TRANSFORMER_LAYER="LlamaDecoderLayer"

# Create minimal FSDP config JSON file
FSDP_CONFIG_FILE="fsdp_config_simple.json"
cat > $FSDP_CONFIG_FILE << EOL
{
  "sharding_strategy": "FULL_SHARD",
  "backward_prefetch": "backward_pre",
  "forward_prefetch": true,
  "cpu_offload": {"offload_params": false},
  "mixed_precision": {"param_dtype": "bfloat16", "reduce_dtype": "bfloat16", "buffer_dtype": "bfloat16"},
  "xla": false,
  "sync_module_states": true,
  "use_orig_params": true,
  "state_dict_type": "sharded"
}
EOL

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME \
    --image_folder $DATA_DIR/images \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --eval_dataset_path $DATA_DIR/$PREFERENCE_DATA \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 1024 \
    --bits 16 \
    --full_finetune True \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 20 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --report_to "wandb" \
    --bf16 True \
    --fsdp "full_shard" \
    --fsdp_config $FSDP_CONFIG_FILE \
    --fsdp_transformer_layer_cls_to_wrap $FSDP_TRANSFORMER_LAYER \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --reward_prompt_file "./prompts/robot_reward_prompt.txt" \
    --image_aspect_ratio 'pad' \
    --run_name "$WANDB_NAME"

# Clean up
rm $FSDP_CONFIG_FILE