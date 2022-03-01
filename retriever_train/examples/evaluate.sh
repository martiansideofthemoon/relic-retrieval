#!/bin/sh
#SBATCH --job-name=eval_retriever_49
#SBATCH -o retriever_train/logs/log_eval_49.txt
#SBATCH --time=167:00:00
#SBATCH --partition=m40-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45GB
#SBATCH -d singleton

# Experiment Details :- RoBERTa-base retriever on new RELIC dataset.
# Run Details :- accumulation = 1, batch_size = 1, beam_size = 1, cpus = 3, dataset = relic_preprocessed/left_4_right_4_neg_100, eval_batch_size = 1, eval_frequency_min = 10, global_dense_feature_list = none, gpu = rtx8000-short, learning_rate = 1e-5, master_port = 4999, memory = 45, model_name = roberta-base, negative_examples = suffix, ngpus = 1, num_epochs = 10, optimizer = adam, prefix_input_type = original, prefix_truncate_dir = both, save_steps = 250, save_total_limit = -1, specific_style_train = -1, stop_token = eos

source /mnt/nfs/work1/miyyer/kalpesh/projects/retrieval-lm/.bashrc
export DATA_DIR=relic_preprocessed/left_4_right_4_neg_100

BASE_DIR=retriever_train

python -m torch.distributed.launch --master_port 5006 --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
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
    --job_id 49 \
    --learning_rate 1e-5 \
    --prefix_input_type original \
    --global_dense_feature_list none \
    --specific_style_train -1 \
    --eval_frequency_min 10 \
    --negative_examples suffix \
    --prefix_truncate_dir both

