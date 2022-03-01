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

## Preprocess Dataset

(coming soon)

## Training dense-ReLIC

### Hyperparameters and Configurations

## Running Baselines (DPR, BM25, SIM, c-REALM, ColBERT)
