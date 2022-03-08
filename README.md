# RELiC: Retrieving Evidence for Literary Claims

This is the official codebase accompanying our ACL 2022 paper "RELiC: Retrieving Evidence for Literary Claims" (https://openreview.net/forum?id=xcelRQScTjP).

## Setup

The code uses PyTorch 1.10+ and HuggingFace's [`transformers`](https://github.com/huggingface/transformers) library for training the RoBERTa models. To install PyTorch, look for the Python package compatible with your local CUDA setup [here](https://pytorch.org).

```
virtualenv relic-venv
source relic-venv/bin/activate
pip install torch torchvision # currently, this is the version compatible with CUDA 10.1
pip install transformers
pip install tensorboardX
pip install --editable .
```

## Pretrained dense-ReLIC models

(coming soon)


## Evaluation

It's best to run this on a GPU, since dense vectors need to be computed before retrieval takes place.

```
python scripts/relic_evaluation.py --split test --model retriever_train/saved_models/model_denserelic_4_4
```

### Running Baselines (DPR, BM25, SIM, c-REALM, ColBERT)

Additional libraries will be needed to run the baseline retrievers.

```
# for SIM
pip install nltk
pip install sentencepiece

# for c-REALM

```

For c-REALM, download the checkpoint from [here](https://storage.googleapis.com/rt-checkpoint/retriever.zip).

## Training dense-ReLIC

### Preprocess Dataset

Download the dataset from [this link](https://drive.google.com/drive/folders/1A-UhzFdeLiEuTa6cvwSmHKMc1gSBvEGB?usp=sharing), and place the files inside the `data` folder. Your `data` folder should look like,

```
(relic-venv) kalpesh@node187:relic-retrieval$ ls data/
test.json  train.json  val.json
(relic-venv) kalpesh@node187:relic-retrieval$
```

Next, run the following preprocessing script (adjust the `left_sents` / `right_sents` flags for shorter contexts):

```
python scripts/preprocess_lit_analysis_data.py --left_sents 4 --right_sents 4
```

### Training and early stopping evaluation

Two scripts are used while training dense-ReLIC, a model training script and an early stopping evaluation script. Both scripts can be run simultaneously --- the evaluation script periodically looks at the checkpoint folder and deletes suboptimal checkpoints. Alternatively, the evaluation script can be run after the model training is finished (to find the best checkpoints).

There are two ways to run training (directly or using SLURM) ---

1. Run the example bash scripts directly,

```
# in terminal 1
# you may need to run "export CUDA_VISIBLE_DEVICES=0" to use GPU-0
bash retriever_train/examples/schedule.sh

# in terminal 2
# you may need to run "export CUDA_VISIBLE_DEVICES=1" to use GPU-1
# this script is used for early stopping checkpoints, it is not a precise evaluation. See section on evaluation for precise evaluation.
bash retriever_train/examples/evaluate.sh
```

2. (recommended) If you have a SLURM setup, you can configure model hyperparameters using [`retriever_train/hyperparameter_config.py`](retriever_train/hyperparameter_config.py) (which supports grid search too) and then run,

```
python retriever_train/schedule.py
```

This script launches both train / evaluation processes simultaneously on SLURM giving them a unique job_id (let's say `X`). You can access the logs using,

```
### Access training logs
cat retriever_train/logs/log_X.txt

### Access early stopping evaluation logs
cat retriever_train/logs/log_eval_X.txt

### Access hyperparameter config for experiment X
cat retriever_train/logs/expts.txt | grep -A "model_X"

### Access the bash scripts running on SLURM
cat retriever_train/slurm-schedulers/schedule_X.sh
cat retriever_train/slurm-schedulers/evaluate_X.sh
```

This script exports checkpoints to [`retriever_train/saved_models/model_X`](retriever_train/saved_models/model_X). There's also TensorBoard support, see [`retriever_train/runs`](retriever_train/runs).

*NOTE*: You may need to make minor changes to [`retriever_train/run_finetune_gpt2_template.sh`](retriever_train/run_finetune_gpt2_template.sh), [`retriever_train/run_evaluate_gpt2_template.sh`](retriever_train/run_evaluate_gpt2_template.sh) and [`retriever_train/schedule.py`](retriever_train/schedule.py) to make them compatible with your SLURM setup.

