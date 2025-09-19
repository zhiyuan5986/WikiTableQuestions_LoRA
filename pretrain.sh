#!/bin/bash
# PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# nohup python run_SHA_pretrain.py \
#     --model_name_or_path /home/qiaoan/data/meta-llama-2-7b-chat-hf \
#     --dataset_path /home/qiaoan/data/long-llm-data/redpajama \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 5e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --warmup_ratio 0.1 \
#     --max_grad_norm 2.0 \
#     --max_seq_length 128 \
#     --output_dir output/meta-llama-2-7b-chat-hf/MTP \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_length 5000 \
#     > ./log/SHA_pretrain_llama_lr5e-5.log 2>&1 &


# while true; do
#     if ! nvidia-smi | grep python; then
#     nohup deepspeed run_CHA_pretrain.py \
#     --model_name_or_path /mnt/hdd-storage/qiaoan/data/Llama-3.1-8B-Instruct \
#     --tokenizer_name_or_path /mnt/hdd-storage/qiaoan/data/Llama-3.1-8B-Instruct \
#     --dataset_path /mnt/hdd-storage/qiaoan/data/totto_data/train \
#     --save_path /mnt/hdd-storage/qiaoan/data/totto_data/train/CHA_pretrain_llama_3200 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 5e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --fp16 false \
#     --output_dir /mnt/hdd-storage/qiaoan/data/CHA_output/Llama-3.1-8B-Instruct/CHA-pretrain-llama-gradient32-time20250903113813-localrank0 \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_length 3200 \
#     --deepspeed configs/ds_config_zero2.json \
#     > ./log/CHA_pretrain_llama_3200_totto.log 2>&1 &
#     break
#     fi
#     echo "Waiting for GPU to be available..."
#     sleep 300  # Wait for 5 minutes
# done

# while true; do
#     if ! nvidia-smi | grep python; then
#     nohup python run_CHA_pretrain.py \
#     --model_name_or_path /mnt/hdd-storage/qiaoan/data/Meta-Llama-3-8B-Instruct \
#     --tokenizer_name_or_path /mnt/hdd-storage/qiaoan/data/Meta-Llama-3-8B-Instruct \
#     --dataset_path /mnt/hdd-storage/qiaoan/data/totto_data/train \
#     --save_path /mnt/hdd-storage/qiaoan/data/totto_data/train/CHA_pretrain_llama3_3200 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 5e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --fp16 false \
#     --output_dir /mnt/hdd-storage/qiaoan/data/CHA_LoRA_output/Meta-Llama-3-8B-Instruct/CHA-LoRA-pretrain \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_length 3200 \
#     --lora_r 32 \
#     --lora_dropout 0.1 \
#     > ./log/CHA_pretrain_llama3_3200_totto.log 2>&1 &
#     break
#     fi
#     echo "Waiting for GPU to be available..."
#     sleep 300  # Wait for 5 minutes
# done

# while true; do
#     if ! nvidia-smi | grep python; then
    nohup python run_CHA_pretrain.py \
    --model_name_or_path /mnt/hdd-storage/qiaoan/data/Llama-3.1-8B-Instruct \
    --tokenizer_name_or_path /mnt/hdd-storage/qiaoan/data/Llama-3.1-8B-Instruct \
    --dataset_path /mnt/hdd-storage/qiaoan/data/totto_data/train \
    --save_path /mnt/hdd-storage/qiaoan/data/totto_data/train/CHA_pretrain_llama_3200 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-5 \
    --remove_unused_columns false \
    --do_train true \
    --do_eval false \
    --seed 42 \
    --bf16 true \
    --fp16 false \
    --output_dir /mnt/hdd-storage/qiaoan/data/CHA_LoRA_output/Llama-3.1-8B-Instruct/CHA-LoRA-pretrain-lorar-128-llama3.1-gradient32-time20250919101635-localrank0 \
    --save_steps 10 \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --logging_strategy steps \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --use_cpu false \
    --low_cpu_mem_usage false \
    --max_length 3200 \
    --lora_r 128 \
    --lora_dropout 0.1 \
    > ./log/CHA_pretrain_llama3.1_3200_totto_lorar_128.log 2>&1 &
#     break
#     fi
#     echo "Waiting for GPU to be available..."
#     sleep 300  # Wait for 5 minutes
# done

# nohup deepspeed run_CHA_pretrain.py \
#     --model_name_or_path /home/qiaoan/data/deepseek-coder-7b-base-v1.5 \
#     --tokenizer_name_or_path /home/qiaoan/data/deepseek-coder-7b-base-v1.5 \
#     --dataset_path /home/qiaoan/data/totto_data/train \
#     --save_path /home/qiaoan/data/totto_data/train/CHA_pretrain_deepseek_3200 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 5e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --fp16 false \
#     --output_dir output/deepseek-coder-7b-base-v1.5/CHA-pretrain \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_table_length 3200 \
#     --deepspeed configs/ds_config_zero2.json \
#     > ./log/CHA_pretrain_deepseek_3200_totto.log 2>&1 &

# while true; do
#     if ! nvidia-smi | grep python; then
#     nohup deepspeed run_CHA_pretrain.py \
#     --model_name_or_path /mnt/hdd-storage/qiaoan/data/deepseek-coder-7b-instruct-v1.5 \
#     --tokenizer_name_or_path /mnt/hdd-storage/qiaoan/data/deepseek-coder-7b-instruct-v1.5 \
#     --dataset_path /mnt/hdd-storage/qiaoan/data/totto_data/train \
#     --save_path /mnt/hdd-storage/qiaoan/data/totto_data/train/CHA_pretrain_deepseek_3200 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 5e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --fp16 false \
#     --output_dir /mnt/hdd-storage/qiaoan/data/CHA_output/deepseek-coder-7b-instruct-v1.5/CHA-pretrain \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_length 3200 \
#     --deepspeed configs/ds_config_zero2.json \
#     > ./log/CHA_pretrain_deepseek_3200_totto.log 2>&1 &
#     break
#     fi
#     echo "Waiting for GPU to be available..."
#     sleep 300  # Wait for 5 minutes
# done

# nohup deepspeed run_CHA_pretrain.py \
#     --model_name_or_path /home/qiaoan/data/Mistral-7B-v0.3 \
#     --tokenizer_name_or_path /home/qiaoan/data/Mistral-7B-v0.3 \
#     --dataset_path /home/qiaoan/data/totto_data/train \
#     --save_path /home/qiaoan/data/totto_data/train/CHA_pretrain_mistral_3200 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 5e-5 \
#     --remove_unused_columns false \
#     --do_train true \
#     --do_eval false \
#     --seed 42 \
#     --bf16 true \
#     --fp16 false \
#     --output_dir output/Mistral-7B-v0.3/CHA-pretrain \
#     --save_steps 10 \
#     --gradient_checkpointing true \
#     --torch_dtype bfloat16 \
#     --attn_implementation eager \
#     --logging_strategy steps \
#     --logging_steps 1 \
#     --num_train_epochs 1 \
#     --use_cpu false \
#     --low_cpu_mem_usage false \
#     --max_length 3200 \
#     --deepspeed configs/ds_config_zero2.json \
#     > ./log/CHA_pretrain_mistral_3200_totto.log 2>&1 &