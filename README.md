# RELiC: Retrieving Evidence for Literary Claims

This is the official codebase accompanying our ACL 2022 paper "RELiC: Retrieving Evidence for Literary Claims" (https://openreview.net/forum?id=xcelRQScTjP).

## Setup

The code uses PyTorch 1.4+ and HuggingFace's [`transformers`](https://github.com/huggingface/transformers) library for training the RoBERTa models. To install PyTorch, look for the Python package compatible with your local CUDA setup [here](https://pytorch.org).

```
virtualenv relic-venv
source relic-venv/bin/activate
pip install torch torchvision # currently, this is the version compatible with CUDA 10.1
pip install transformers
pip install --editable .
```

## Pretrained dense-ReLIC models

(coming soon)

## Preprocess Dataset

(coming soon)

## Training dense-ReLIC

Two scripts are used while training dense-ReLIC, a model training script and an evaluation script. Both scripts can be run simultaneously --- the evaluation script periodically looks at the checkpoint folder and deletes suboptimal checkpoints. Alternatively, the evaluation script can be run after the model training is finished (to find the best checkpoints).

There are two ways to run training ---

1. Run the example bash scripts directly,

```
# in terminal 1
# you may need to run "export CUDA_VISIBLE_DEVICES=0" to use GPU-0
retriever_train/examples/schedule.sh

# in terminal 2
# you may need to run "export CUDA_VISIBLE_DEVICES=1" to use GPU-1
retriever_train/examples/evaluate.sh
```

2. If you have a SLURM setup, you can configure model hyperparameters using [`retriever_train/hyperparameter_config.py`](retriever_train/hyperparameter_config.py) (which supports grid search too) and then run,

```
python retriever_train/schedule.py
```

This script launches both train / evaluation processes simultaneously on SLURM giving them a unique job_id (let's say `X`), adds a summary to [`retriever_train/logs/expts.txt`](retriever_train/logs/expts.txt), export checkpoints to [`retriever_train/saved_models/model_X`](retriever_train/saved_models/model_X), training logs to [`retriever_train/logs/log_X.txt`](retriever_train/logs/log_X.txt) and evaluation logs to [`retriever_train/logs/log_eval_X.txt`](retriever_train/logs/log_eval_X.txt). There's also TensorBoard support, see [`retriever_train/runs`](retriever_train/runs).

You may need to make minor changes to [`retriever_train/run_finetune_gpt2_template.sh`](retriever_train/run_finetune_gpt2_template.sh), [`retriever_train/run_evaluate_gpt2_template.sh`](retriever_train/run_evaluate_gpt2_template.sh) and [`retriever_train/schedule.py`](retriever_train/schedule.py) to make them compatible with your SLURM setup.

## Running Baselines (DPR, BM25, SIM, c-REALM, ColBERT)
