# export HF_ENDPOINT=https://hf-mirror.com

wandb disabled

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "发现 ${NUM_GPUS} 块 GPU。"
else
    NUM_GPUS=-1
    echo "nvidia-smi 命令不可用，请确保 NVIDIA 驱动已安装。"
fi


model_name=qwen
model_path=/mnt/data/hf_models/Qwen2-7B-Instruct

dataset_name=weak
dataset_path=data/${dataset_name}

config_file=deepspeed_zero2

tag=${dataset_name}_${model_name}_${config_file}

echo "dataset_name ${dataset_name}, tag ${tag}"

accelerate launch --config_file accelerate_configs/${config_file}.yaml --num_processes ${NUM_GPUS} dpo.py \
    --dataset_name ${dataset_path} \
    --model_name_or_path ${model_path} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --logging_steps 3 \
    --eval_steps 5 \
    --output_dir output_model/${tag} \
    --warmup_steps 5 \
    --report_to none \
    --bf16 \
    --logging_first_step \
    --max_prompt_length 512 \
    --max_length 512 \
    --no_remove_unused_columns > log/train_log/${tag}.log 2>&1

echo "Done."
