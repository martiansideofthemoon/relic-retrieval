#!/bin/sh
#SBATCH --job-name=finetune_retriever_{job_id}
#SBATCH -o retriever_train/logs/log_{job_id}.txt
#SBATCH --time={time}:00:00
#SBATCH --partition={gpu}
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

source relic-venv/bin/activate
export DATA_DIR={dataset}

BASE_DIR=retriever_train

# Snapshot code used for the run
mkdir -p $BASE_DIR/saved_models/model_{job_id}_code

cp $BASE_DIR/*.py $BASE_DIR/saved_models/model_{job_id}_code

echo $HOSTNAME

torchrun --master_port {master_port} --nproc_per_node={ngpus} $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_{job_id} \
    --model_type=roberta \
    --model_name_or_path={model_name} \
    --do_train \
    --data_dir=$DATA_DIR \
    --save_steps {save_steps} \
    --logging_steps 20 \
    --save_total_limit {save_total_limit} \
    --evaluate_during_training \
    --num_train_epochs {num_epochs} \
    --gradient_accumulation_steps {accumulation} \
    --per_gpu_train_batch_size {batch_size} \
    --job_id {job_id} \
    --learning_rate {learning_rate} \
    --optimizer {optimizer} \
    --negative_examples {negative_examples} \
    --prefix_truncate_dir {prefix_truncate_dir}
