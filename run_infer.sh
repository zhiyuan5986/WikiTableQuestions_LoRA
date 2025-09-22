PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_PATH='/mnt/hdd-storage/qiaoan/data/TableBench'
SAVE_PATH='/home/qiaoan/Documents/WikiTableQuestions_LoRA/eval_examples/inference_results'
MODEL_DIR='/mnt/hdd-storage/qiaoan/data/Llama-3.1-8B-Instruct'
PARAM_DIR='/mnt/hdd-storage/qiaoan/data/CHA_LoRA_output/Llama-3.1-8B-Instruct/CHA-LoRA-finetune-lorar-128-llama3.1-gradient32-epochs1.0-time20250921171840-localrank0'
TOKENIZER_DIR='/mnt/hdd-storage/qiaoan/data/Llama-3.1-8B-Instruct'

while true; do
    if ! nvidia-smi | grep pythonn; then
    CUDA_LAUNCH_BLOCKING=1 python infer.py \
        --model_name_or_path $MODEL_DIR \
        --param_dir $PARAM_DIR \
        --tokenizer_name_or_path $TOKENIZER_DIR \
        --dataset_path $DATA_PATH \
        --save_path $SAVE_PATH \
        --torch_dtype bfloat16 \
        --attn_implementation eager \
        --low_cpu_mem_usage true 
    break
    fi
    echo "Waiting for GPU to be available..."
    sleep 300
done