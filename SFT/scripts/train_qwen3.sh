#!/bin/bash
export TORCH_CUDA_ARCH_LIST="10.0"


llm_versions=("Qwen3-VL-4B-Instruct" "Qwen3-VL-8B-Instruct")
llm_paths=("Qwen/Qwen3-VL-4B-Instruct" "Qwen/Qwen3-VL-8B-Instruct")


export ACCELERATE_CPU_AFFINITY=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29504
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


num_epochs=1
use_action_weight=False
action_weight=1
for llm_index in  0  1
do
    LLM_VERSION=${llm_versions[llm_index]}
    LLM_PATH=${llm_paths[llm_index]}
    for mode in 'reasoning_and_grounding_changecoord_mixnoreasoning_qwen3' 
    do
        SFT_TASK="${mode}"
        SAVE_DIR=checkpoints
        IMAGE_FOLDER=./data/images

        SFT_DATA_YAML=data/${SFT_TASK}.yaml
        SFT_RUN_NAME="${LLM_VERSION}-sft-${SFT_TASK}"_useweight${use_action_weight}_weight${action_weight}_lr1e-5_coord1000
        echo "SFT_RUN_NAME: ${SFT_RUN_NAME}"

        printenv

        accelerate launch --num_processes 8 train.py \
            --data_path ${SFT_DATA_YAML} \
            --image_folder ${IMAGE_FOLDER} \
            --model_name_or_path $LLM_PATH \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir ${SAVE_DIR}/${SFT_RUN_NAME} \
            --num_train_epochs ${num_epochs} \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 8 \
            --attn_implementation "flash_attention_2" \
            --eval_strategy "no" \
            --save_strategy "epoch" \
            --save_steps 100 \
            --save_total_limit 5 \
            --learning_rate 1e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.01 \
            --lr_scheduler_type "cosine" \
            --use_action_weight ${use_action_weight} \
            --action_weight ${action_weight} \
            --logging_steps 10 \
            --tf32 True \
            --model_max_length 24576 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --freeze_visual_encoder False \
            --report_to "wandb" 
    done
done


