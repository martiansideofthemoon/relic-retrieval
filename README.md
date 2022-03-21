# RELiC: Retrieving Evidence for Literary Claims

This is the official codebase accompanying our ACL 2022 paper "RELiC: Retrieving Evidence for Literary Claims" (https://relic.cs.umass.edu). You can find our paper on arXiv [here](https://arxiv.org/abs/2203.10053).

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

**Download the dataset** from [this link](https://drive.google.com/drive/folders/1A-UhzFdeLiEuTa6cvwSmHKMc1gSBvEGB?usp=sharing). Your `RELiC` folder should look like,

```
(relic-venv) kalpesh@node187:relic-retrieval$ ls RELiC/
test.json  train.json  val.json
(relic-venv) kalpesh@node187:relic-retrieval$
```

## Pretrained dense-RELiC models

All pretrained models can be found in the dataset Google Drive [folder](https://drive.google.com/drive/folders/1A-UhzFdeLiEuTa6cvwSmHKMc1gSBvEGB?usp=sharing). Individual checkpoint links are added below,

| Model                                   | Google Drive link |
|-----------------------------------------|-------------------|
| **dense-RELiC (4 left, 4 right sentences)** | [link](https://drive.google.com/drive/folders/1Y2PxHRycWucQtQCbw5OxN8eQMEZtS43h?usp=sharing)              |
| dense-RELiC (4 left, 0 right sentences) | [link](https://drive.google.com/drive/folders/1TfKMY-XZhI5IVXPpRZ59n3j2jiw_VQcH?usp=sharing)              |
| dense-RELiC (0 left, 4 right sentences) | [link](https://drive.google.com/drive/folders/1fw9BRrNnU9TzkabBf7PUrWyzXQCqFrB0?usp=sharing)              |
| dense-RELiC (1 left, 1 right sentences) | [link](https://drive.google.com/drive/folders/1GFgwXyEYg0IL5eYJ-sqg9Zm2hKTVXOFt?usp=sharing)              |
| dense-RELiC (1 left, 0 right sentences) | [link](https://drive.google.com/drive/folders/1us5uy_aRl4StUxD1MzZEecB4fH--6R2e?usp=sharing)              |

## Evaluation

Make sure you have downloaded the dataset as described [above](#setup). The evaluation script assumes the pretrained models are downloaded from the Google Drive links above and placed in the `retriever_train/saved_models`. It's best to run this on a GPU, since dense vectors need to be computed before retrieval takes place.

```
# you may need to run "export CUDA_VISIBLE_DEVICES=0" to use GPU-0
# remove --cache if you don't wish to write a large output file with retrieval ranks
python scripts/relic_evaluation.py \
    --model retriever_train/saved_models/model_denserelic_4_4 \
    --cache

# output
Results with all quotes (7833 instances):
mean_rank = 704.6351, recall@1 = 0.0672, recall@3 = 0.1407, recall@5 = 0.1840, recall@10 = 0.2578, recall@50 = 0.4501, recall@100 = 0.5361, num_candidates = 10199.8426
```

The above script may take a while to finish (20-30 minutes on validation data). To run it on a single book only, run:

```
python scripts/relic_evaluation.py \
    --model retriever_train/saved_models/model_denserelic_4_4 \
    --eval_small

# output
Results with all quotes (1648 instances):
mean_rank = 796.5661, recall@1 = 0.0583, recall@3 = 0.1147, recall@5 = 0.1481, recall@10 = 0.2093, recall@50 = 0.3914, recall@100 = 0.4745, num_candidates = 9775.8471
```

## Training dense-RELiC

### Preprocess Dataset

Make sure you have downloaded the dataset as described [above](#setup). Run the following preprocessing script (adjust the `--left_sents` / `--right_sents` flags for shorter contexts):

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
# this script is used for early stopping checkpoints, it is not a precise evaluation.
bash retriever_train/examples/evaluate.sh
```

2. If you have a SLURM setup, you can configure model hyperparameters using [`retriever_train/hyperparameter_config.py`](retriever_train/hyperparameter_config.py) (which supports grid search too) and then run,

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

## Running Baselines (DPR, SIM, c-REALM)

Additional libraries will be needed to run the baseline retrievers.

1. **SIM** --- A semantic similarity model from [Wieting et al. 2019](https://aclanthology.org/P19-1427) trained on STS data.

```
pip install nltk
pip install sentencepiece

# remove --cache if you don't wish to write a large output file with retrieval ranks
python scripts/relic_evaluation_sim.py --left_sents 1 --right_sents 1 --cache
```

2. **DPR** --- A retriever from [Karphukin et al. 2020](https://aclanthology.org/2020.emnlp-main.550) trained on [Natural Questions](https://ai.google.com/research/NaturalQuestions) data.

```
# remove --cache if you don't wish to write a large output file with retrieval ranks
python scripts/relic_evaluation_dpr.py --left_sents 1 --right_sents 1 --cache
```

3. **c-REALM** --- A retriever from [Krishna et al. 2021](https://aclanthology.org/2021.naacl-main.393) based on [REALM](https://arxiv.org/abs/2002.08909) and trained on [ELI5](https://arxiv.org/abs/1907.09190) data.

```
### for c-REALM
# TF 2.3 is the version compatible with CUDA 10.1
# See https://www.tensorflow.org/install/source#gpu for TF-CUDA mapping
pip install tensorflow==2.3
pip install tensor2tensor

# Download and unzip the c-REALM checkpoint
wget https://storage.googleapis.com/rt-checkpoint/retriever.zip
unzip retriever.zip && rm retriever.zip
mv retriever crealm-retriever
rm -rf crealm-retriever/encoded_*

# remove --cache if you don't wish to write a large output file with retrieval ranks
python scripts/relic_evaluation_crealm.py --left_sents 1 --right_sents 1 --cache
```

4. **Random retrieval**

```
python scripts/relic_evaluation_random.py  --num_samples 100 --split val
```

## Leaderboard Submission

You may submit your predictions for the test set here: https://forms.gle/1B6JuQ3nbGXCR2kC8. The leaderboard is maintained on the RELiC project page [here](https://relic.cs.umass.edu/leaderboard.html).

The format of your submission file should be a `.json` file that is a dictionary where the unique IDs of each test set quote are the keys, and the values are a rank list. This list should contain the 100 indices of the top 100 candidates retriever by your model, in rank order. For example, if your retriever's top-ranked candidate is `99` for test set quote ID `"070789"` and `1532` for quote ID `"070790"`, your `.json` dict should look like:

``` json
{
    "070789": [99, ...],
    "070790": [1532, ...],
    ...
}
```

To make this file with dense-RELiC (or any of our other baselines), run the corresponding evaluation script with the `--split test` flag:

```
python scripts/relic_evaluation.py \
    --model retriever_train/saved_models/model_denserelic_4_4 \
    --split test
```

This will output a file `retriever_train/saved_models/model_denserelic_4_4/test_submission.json`, which you should upload to the Google Form. We will read this JSON file using [`scripts/score_submission.py`](scripts/score_submission.py) using a hidden key file, and upload the results on the leaderboard.


## Contact & Citation

If you run into any issues, please contact [kbthai@umass.edu](mailto:kbthai@umass.edu) and [kalpesh@cs.umass.edu](mailto:kalpesh@cs.umass.edu).

If you found our paper or this repository useful, please cite:

```
@inproceedings{relic22,
author={Katherine Thai and Yapei Chang and Kalpesh Krishna and Mohit Iyyer},
Booktitle = {Association of Computational Linguistics},
Year = "2022",
Title={RELiC: Retrieving Evidence for Literary Claims},
}
```
