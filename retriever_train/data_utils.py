import logging
import numpy as np
import pickle
import random

MAX_ROBERTA_LENGTH = 502

random.seed(12)
logger = logging.getLogger(__name__)


class Instance(object):
    def __init__(self, args, config, instance, negative_examples=[]):
        self.args = args
        self.config = config
        self.either_truncated = 0.0
        self.prefix_truncated = 0.0
        self.suffix_truncated = 0.0
        self.instance = instance
        self.negative_examples = negative_examples

    def preprocess(self, tokenizer):
        max_prefix_length = self.config["max_prefix_length"]
        max_suffix_length = self.config["max_suffix_length"]

        self.prefices = []
        self.prefix_masks = []
        self.suffices = []
        self.suffix_masks = []

        self.negatives = []
        self.negative_masks = []

        for kk, (prefix, suffix) in enumerate(self.instance):
            if len(prefix) > max_prefix_length:
                self.prefix_truncated += 1 / len(self.instance)
            if len(suffix) > max_suffix_length:
                self.suffix_truncated += 1 / len(self.instance)
            if len(prefix) > max_prefix_length or len(suffix) > max_suffix_length:
                self.either_truncated += 1 / len(self.instance)

            if len(prefix) >= max_prefix_length:
                if self.args.prefix_truncate_dir == "left":
                    prefix = [prefix[0]] + prefix[-max_prefix_length + 1:]
                elif self.args.prefix_truncate_dir == "right":
                    prefix = prefix[:max_prefix_length]
                elif self.args.prefix_truncate_dir == "both":
                    # keep only span around <mask> token
                    mask_pos = prefix.index(tokenizer.mask_token_id)
                    # Assume prefix length is even number
                    assert max_prefix_length % 2 == 0
                    left_window_pos = mask_pos - max_prefix_length // 2
                    right_window_pos = mask_pos + max_prefix_length // 2
                    # Make sure that truncation window is within bounds in at least in one direction
                    assert left_window_pos >= 0 or right_window_pos <= len(prefix)
                    if left_window_pos < 0:
                        left_window_pos = 0
                        right_window_pos = max_prefix_length
                    elif right_window_pos > len(prefix):
                        right_window_pos = len(prefix)
                        left_window_pos = len(prefix) - max_prefix_length
                    prefix = prefix[left_window_pos:right_window_pos]
                    assert tokenizer.mask_token_id in prefix and len(prefix) == max_prefix_length

                self.prefix_masks.append([1 for _ in range(max_prefix_length)])
            else:
                prefix = right_padding(prefix, tokenizer.pad_token_id, max_prefix_length)
                self.prefix_masks.append(
                    [1 for _ in range(len(prefix))] + [0 for _ in range(max_prefix_length - len(prefix))]
                )
            self.prefices.append(prefix)

            if len(suffix) >= max_suffix_length:
                suffix = suffix[:max_suffix_length]
                self.suffix_masks.append([1 for _ in range(max_suffix_length)])
            else:
                suffix = right_padding(suffix, tokenizer.pad_token_id, max_suffix_length)
                self.suffix_masks.append(
                    [1 for _ in range(len(suffix))] + [0 for _ in range(max_suffix_length - len(suffix))]
                )
            self.suffices.append(suffix)

            if len(self.negative_examples) == 0:
                continue
            curr_negative = self.negative_examples[kk]
            if len(curr_negative) >= max_suffix_length:
                curr_negative = curr_negative[:max_suffix_length]
                self.negative_masks.append([1 for _ in range(max_suffix_length)])
            else:
                curr_negative = right_padding(curr_negative, tokenizer.pad_token_id, max_suffix_length)
                self.negative_masks.append(
                    [1 for _ in range(len(curr_negative))] + [0 for _ in range(max_suffix_length - len(curr_negative))]
                )
            self.negatives.append(curr_negative)

        for pf, pf_mask, sf, sf_mask in zip(self.prefices, self.prefix_masks, self.suffices, self.suffix_masks):
            assert len(pf) == max_prefix_length
            assert len(pf_mask) == max_prefix_length
            assert len(sf) == max_suffix_length
            assert len(sf_mask) == max_suffix_length

        self.prefices = np.array(self.prefices)
        self.prefix_masks = np.array(self.prefix_masks)
        self.suffices = np.array(self.suffices)
        self.suffix_masks = np.array(self.suffix_masks)
        self.negatives = np.array(self.negatives)
        self.negative_masks = np.array(self.negative_masks)


def np_prepend(array, value):
    return np.insert(array, 0, value)


def left_padding(data, pad_token, total_length):
    tokens_to_pad = total_length - len(data)
    return np.pad(data, (tokens_to_pad, 0), constant_values=pad_token)


def right_padding(data, pad_token, total_length):
    tokens_to_pad = total_length - len(data)
    return np.pad(data, (0, tokens_to_pad), constant_values=pad_token)


def string_to_ids(text, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def get_label_dict(data_dir):
    label_dict = {}
    with open("{}/dict.txt".format(data_dir)) as f:
        label_dict_lines = f.read().strip().split("\n")
    for i, x in enumerate(label_dict_lines):
        if x.startswith("madeupword"):
            continue
        label_dict[x.split()[0]] = i
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    return label_dict, reverse_label_dict


def get_global_dense_features(data_dir, global_dense_feature_list, label_dict):
    """Get dense style code vectors for the style code model."""

    global_dense_features = []
    if global_dense_feature_list != "none":
        logger.info("Using global dense vector features = %s" % global_dense_feature_list)
        for gdf in global_dense_feature_list.split(","):
            with open("{}/{}_dense_vectors.pickle".format(data_dir, gdf), "rb") as f:
                vector_data = pickle.load(f)

            final_vectors = {}
            for k, v in vector_data.items():
                final_vectors[label_dict[k]] = v["sum"] / v["total"]

            global_dense_features.append((gdf, final_vectors))
    return global_dense_features


def limit_dataset_size(dataset, limit_examples):
    """Limit the dataset size to a small number for debugging / generation."""

    if limit_examples:
        logger.info("Limiting dataset to {:d} examples".format(limit_examples))
        dataset = dataset[:limit_examples]

    return dataset


def limit_styles(dataset, specific_style_train, split, reverse_label_dict):
    """Limit the dataset size to a certain author."""
    specific_style_train = [int(x) for x in specific_style_train.split(",")]

    original_dataset_size = len(dataset)
    if split in ["train", "test"] and -1 not in specific_style_train:
        logger.info("Preserving authors = {}".format(", ".join([reverse_label_dict[x] for x in specific_style_train])))
        dataset = [x for x in dataset if x["suffix_style"] in specific_style_train]
        logger.info("Remaining instances after author filtering = {:d} / {:d}".format(len(dataset), original_dataset_size))
    return dataset


def datum_to_dict(datum, tokenizer):
    """Convert a data point to the instance dictionary."""
    all_sents = []
    negative_examples = []
    for pair in datum:
        first_sent = tokenizer(" ".join(pair[0].split()))["input_ids"]
        second_sent = tokenizer(" ".join(pair[1].split()))["input_ids"]
        all_sents.append([first_sent, second_sent])
        if len(pair) == 4:
            negative_sent = tokenizer(" ".join(pair[3].split()))["input_ids"]
            negative_examples.append(negative_sent)

    return all_sents, negative_examples


def update_config(args, config):
    if args.global_dense_feature_list != "none":
        global_dense_length = len(args.global_dense_feature_list.split(","))
        logger.info("Using {:d} dense feature vectors.".format(global_dense_length))
    else:
        global_dense_length = 0

    assert global_dense_length <= config["max_dense_length"]
    config["global_dense_length"] = global_dense_length
