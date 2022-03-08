#!/bin/sh
#SBATCH --job-name=eval_retriever_4_4
#SBATCH -o retriever_train/logs/log_eval_4_4.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45GB
#SBATCH -d singleton

source relic-venv/bin/activate
export DATA_DIR=relic_preprocessed/left_4_right_4_neg_100

BASE_DIR=retriever_train

echo $HOSTNAME

# WARNING --- don't report these numbers in a paper, this script is for early stopping only! Use scripts/relic_evaluation.py instead, see README for instructions.

torchrun --master_port 5006 --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_4_4 \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --data_dir=$DATA_DIR \
    --do_eval \
    --do_delete_old \
    --save_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 3 \
    --evaluate_during_training \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --per_gpu_train_batch_size 1 \
    --limit_examples 1000 \
    --job_id 4_4 \
    --learning_rate 1e-5 \
    --eval_frequency_min 10 \
    --negative_examples suffix \
    --prefix_truncate_dir both

