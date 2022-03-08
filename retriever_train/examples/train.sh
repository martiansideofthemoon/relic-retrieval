#!/bin/sh
#SBATCH --job-name=finetune_retriever_49
#SBATCH -o retriever_train/logs/log_49.txt
#SBATCH --time=4:00:00
#SBATCH --partition=rtx8000-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=45GB
#SBATCH -d singleton

source relic-venv/bin/activate
export DATA_DIR=relic_preprocessed/left_4_right_4_neg_100

BASE_DIR=retriever_train

echo $HOSTNAME

torchrun --master_port 4999 --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_4_4 \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --data_dir=$DATA_DIR \
    --save_steps 250 \
    --logging_steps 20 \
    --save_total_limit -1 \
    --evaluate_during_training \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 1 \
    --job_id 4_4 \
    --learning_rate 1e-5 \
    --optimizer adam \
    --negative_examples suffix \
    --prefix_truncate_dir both

