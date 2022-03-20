import pickle
import torch

import numpy as np

from transformers import RobertaModel, RobertaTokenizerFast

from retriever_train.dataset_config import DATASET_CONFIG, BASE_CONFIG
from retriever_train.data_utils import Instance

from retriever_train.utils import init_parent_model


class PrefixSuffixWrapper(object):
    def __init__(self, model_path, config_only=False):
        self.model_path = model_path
        self.args = torch.load("{}/training_args.bin".format(self.model_path))

        if "prefix_truncate_dir" not in self.args:
            # hack for backward compatability
            self.args.prefix_truncate_dir = "left"

        self.args.device = torch.cuda.current_device()
        if self.args.data_dir in DATASET_CONFIG:
            self.config = DATASET_CONFIG[self.args.data_dir]
        else:
            self.config = BASE_CONFIG
        print(self.config)

        if not config_only:
            self.model, self.tokenizer = init_parent_model(checkpoint_dir=model_path,
                                                           args=self.args,
                                                           model_class=RobertaModel,
                                                           tokenizer_class=RobertaTokenizerFast)

    def preprocess_sentences(self, contexts, vectors_type="prefix"):
        args = self.args
        tokenizer = self.tokenizer
        instances = []

        all_context_ids = []
        for context in contexts:
            context = " ".join(context.split())
            context_ids = tokenizer(context)["input_ids"]
            if vectors_type == "suffix":
                placeholder_prefix_ids = tokenizer("left context <mask> right context")["input_ids"]
                all_context_ids.append([placeholder_prefix_ids, context_ids])
            else:
                all_context_ids.append([context_ids, context_ids])

        instance = Instance(
            self.args, self.config, all_context_ids
        )
        instance.preprocess(tokenizer)
        return instance

    def encode_batch(self, contexts, vectors_type="prefix"):
        args = self.args

        instance = self.preprocess_sentences(contexts, vectors_type)
        input_tensors = {
            "prefices": torch.tensor(instance.prefices).unsqueeze(0),
            "prefix_masks": torch.tensor(instance.prefix_masks).unsqueeze(0),
            "suffices": torch.tensor(instance.suffices).unsqueeze(0),
            "suffix_masks": torch.tensor(instance.suffix_masks).unsqueeze(0)
        }
        return self.model.get_vectors(input_tensors, vectors_type=vectors_type)
