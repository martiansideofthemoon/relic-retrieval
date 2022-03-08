import logging
import json
import os
import pickle
import random
from functools import partial

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from data_utils import (
    Instance,
    datum_to_dict,
    limit_dataset_size
)
from dataset_config import (
    BASE_CONFIG,
    DATASET_CONFIG
)

logger = logging.getLogger(__name__)


class PrefixSuffixDataset(Dataset):
    def __init__(self, tokenizer, args, evaluate=False, split="train"):
        data_dir = args.data_dir
        self.args = args

        if data_dir in DATASET_CONFIG:
            self.config = DATASET_CONFIG[data_dir]
        else:
            self.config = BASE_CONFIG

        logger.info(self.config)

        self.examples = []

        cached_features_file = os.path.join(
            data_dir, args.model_type + "_cached_lm_" + split
        )
        # Caching is important since it can avoid slow tokenization
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            with open("{}/{}.pkl".format(data_dir, split), "rb") as handle:
                data = pickle.load(handle)

            self.examples = [
                datum_to_dict(datum, tokenizer) for datum in tqdm.tqdm(data)
            ]

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # in case we are using a fraction of the dataset, reduce the size of the dataset here
        self.examples = limit_dataset_size(self.examples, args.limit_examples)

        # convert the dataset into the Instance class
        self.examples = [
            Instance(args, self.config, *datum_dict) for datum_dict in self.examples
        ]

        for instance in tqdm.tqdm(self.examples):
            # perform truncation, padding, label and segment building
            instance.preprocess(tokenizer)

        num_truncated = sum([x.either_truncated for x in self.examples])
        num_prefix_truncated = sum([x.prefix_truncated for x in self.examples])
        num_suffix_truncated = sum([x.suffix_truncated for x in self.examples])
        logger.info(
            "Total truncated instances due to length limit = {:.2f} either, {:.2f} prefix, {:.2f} suffix / {:.2f}".format(
                num_truncated, num_prefix_truncated, num_suffix_truncated, len(self.examples)
            )
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if "negatives" in self.args.negative_examples:
            negatives = self.examples[item].negatives
            negative_masks = self.examples[item].negative_masks
        else:
            negatives = np.zeros(10)
            negative_masks = np.zeros(10)
        return {
            "prefices": self.examples[item].prefices,
            "prefix_masks": self.examples[item].prefix_masks,
            "suffices": self.examples[item].suffices,
            "suffix_masks": self.examples[item].suffix_masks,
            "negatives": negatives,
            "negative_masks": negative_masks
        }
