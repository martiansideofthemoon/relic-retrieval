#!/bin/sh
#SBATCH --job-name=eval_retriever_{job_id}
#SBATCH -o retriever_train/logs/log_eval_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

source relic-venv/bin/activate
export DATA_DIR={dataset}

BASE_DIR=retriever_train

# WARNING --- don't report these numbers in a paper, this script is for early stopping only! Use scripts/relic_evaluation.py instead, see README for instructions.

torchrun --master_port {master_port_eval} --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_{job_id} \
    --model_type=roberta \
    --model_name_or_path={model_name} \
    --data_dir=$DATA_DIR \
    --do_eval \
    --do_delete_old \
    --save_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 3 \
    --evaluate_during_training \
    --num_train_epochs {num_epochs} \
    --gradient_accumulation_steps {accumulation} \
    --per_gpu_train_batch_size {batch_size} \
    --limit_examples 1000 \
    --job_id {job_id} \
    --learning_rate {learning_rate} \
    --eval_frequency_min {eval_frequency_min} \
    --negative_examples {negative_examples} \
    --prefix_truncate_dir {prefix_truncate_dir}
