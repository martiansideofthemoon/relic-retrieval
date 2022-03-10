import logging
import os
import torch
import torch.nn as nn

import numpy as np

logger = logging.getLogger(__name__)


def recall(sentence, srl_string):
    matches = 0
    for word in sentence.split():
        if word in srl_string:
            matches += 1

    if len(sentence.split()) > 0:
        return float(matches) / len(sentence.split())
    else:
        return 0


def rindex(mylist, myvalue):
    return len(mylist) - mylist[::-1].index(myvalue) - 1


def init_parent_model(checkpoint_dir, args, model_class, tokenizer_class=None):
    """Load a trained model and vocabulary that you have fine-tuned."""

    prefix_dir = os.path.join(checkpoint_dir, "prefix_encoder")
    suffix_dir = os.path.join(checkpoint_dir, "suffix_encoder")

    prefix_encoder = model_class.from_pretrained(prefix_dir)
    prefix_encoder.to(args.device)

    suffix_encoder = model_class.from_pretrained(suffix_dir)
    suffix_encoder.to(args.device)

    if tokenizer_class:
        tokenizer = tokenizer_class.from_pretrained(checkpoint_dir, do_lower_case=args.do_lower_case)
    else:
        tokenizer = None

    return PrefixSuffixModel(args=args, prefix_encoder=prefix_encoder, suffix_encoder=suffix_encoder), tokenizer


class PrefixSuffixModel(nn.Module):
    def __init__(self, args, prefix_encoder, suffix_encoder):
        super(PrefixSuffixModel, self).__init__()
        self.args = args
        self.prefix_encoder = prefix_encoder
        self.suffix_encoder = suffix_encoder
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        args = self.args
        prefix_encoder = self.prefix_encoder
        suffix_encoder = self.suffix_encoder
        prefix_encoder.train()
        suffix_encoder.train()

        prefices = batch["prefices"].to(args.device)
        suffices = batch["suffices"].to(args.device)
        prefix_masks = batch["prefix_masks"].to(args.device)
        suffix_masks = batch["suffix_masks"].to(args.device)

        batch_size, num_pairs, prefix_seq_length = prefices.shape
        _, _, suffix_seq_length = suffices.shape

        prefix_outs = prefix_encoder(
            input_ids=prefices.reshape(-1, prefix_seq_length),
            attention_mask=prefix_masks.reshape(-1, prefix_seq_length)
        )
        prefix_outs = prefix_outs.pooler_output
        suffix_outs = suffix_encoder(
            input_ids=suffices.reshape(-1, suffix_seq_length),
            attention_mask=suffix_masks.reshape(-1, suffix_seq_length)
        )
        suffix_outs = suffix_outs.pooler_output
        positive_negative_vectors = suffix_outs

        if "prefix" in args.negative_examples:
            prefix_outs_suffix_encoder = suffix_encoder(
                input_ids=prefices.reshape(-1, prefix_seq_length),
                attention_mask=prefix_masks.reshape(-1, prefix_seq_length)
            )
            prefix_outs_suffix_encoder = prefix_outs_suffix_encoder.pooler_output
            positive_negative_vectors = torch.cat([suffix_outs, prefix_outs_suffix_encoder], dim=0)

        elif "negatives" in args.negative_examples:
            negatives = batch["negatives"].to(args.device)
            negative_masks = batch["negative_masks"].to(args.device)

            negatives_outs_suffix_encoder = suffix_encoder(
                input_ids=negatives.reshape(-1, suffix_seq_length),
                attention_mask=negative_masks.reshape(-1, suffix_seq_length)
            )
            negatives_outs_suffix_encoder = negatives_outs_suffix_encoder.pooler_output
            positive_negative_vectors = torch.cat([suffix_outs, negatives_outs_suffix_encoder], dim=0)

        similarities = torch.matmul(prefix_outs, positive_negative_vectors.t())
        labels = torch.arange(batch_size * num_pairs).cuda()
        loss = self.loss_fn(similarities, labels)
        loss = {
            "lm": loss
        }

        return loss

    def evaluate(self, batch):

        args = self.args
        prefix_encoder = self.prefix_encoder
        suffix_encoder = self.suffix_encoder
        prefix_encoder.eval()
        suffix_encoder.eval()

        prefices = batch["prefices"].to(args.device)
        suffices = batch["suffices"].to(args.device)
        prefix_masks = batch["prefix_masks"].to(args.device)
        suffix_masks = batch["suffix_masks"].to(args.device)

        batch_size, num_pairs, prefix_seq_length = prefices.shape
        _, _, suffix_seq_length = suffices.shape

        with torch.no_grad():
            prefix_outs = prefix_encoder(
                input_ids=prefices.reshape(-1, prefix_seq_length),
                attention_mask=prefix_masks.reshape(-1, prefix_seq_length)
            )
            prefix_outs = prefix_outs.pooler_output
            suffix_outs = suffix_encoder(
                input_ids=suffices.reshape(-1, suffix_seq_length),
                attention_mask=suffix_masks.reshape(-1, suffix_seq_length)
            )
            suffix_outs = suffix_outs.pooler_output
            positive_negative_vectors = suffix_outs

            if "prefix" in args.negative_examples:
                prefix_outs_suffix_encoder = suffix_encoder(
                    input_ids=prefices.reshape(-1, prefix_seq_length),
                    attention_mask=prefix_masks.reshape(-1, prefix_seq_length)
                )
                prefix_outs_suffix_encoder = prefix_outs_suffix_encoder.pooler_output
                positive_negative_vectors = torch.cat([suffix_outs, prefix_outs_suffix_encoder], dim=0)

            elif "negatives" in args.negative_examples:
                negatives = batch["negatives"].to(args.device)
                negative_masks = batch["negative_masks"].to(args.device)

                negatives_outs_suffix_encoder = suffix_encoder(
                    input_ids=negatives.reshape(-1, suffix_seq_length),
                    attention_mask=negative_masks.reshape(-1, suffix_seq_length)
                )
                negatives_outs_suffix_encoder = negatives_outs_suffix_encoder.pooler_output
                positive_negative_vectors = torch.cat([suffix_outs, negatives_outs_suffix_encoder], dim=0)


            similarities = torch.matmul(prefix_outs, positive_negative_vectors.t())
            labels = torch.arange(batch_size * num_pairs).cuda()
            loss = self.loss_fn(similarities, labels)
            correct = torch.sum(torch.argmax(similarities, dim=1) == labels)

            sorted_scores = torch.argsort(similarities, dim=1, descending=True)
            ranks = [sorted_scores[i].tolist().index(i) + 1 for i in range(batch_size * num_pairs)]

            results = {
                "loss": loss,
                "accuracy": correct.item() / len(labels),
                "mean_rank": np.mean(ranks),
                "R@1": len([x for x in ranks if x <= 1]) / len(ranks),
                "R@3": len([x for x in ranks if x <= 3]) / len(ranks),
                "R@5": len([x for x in ranks if x <= 5]) / len(ranks),
                "R@10": len([x for x in ranks if x <= 10]) / len(ranks),
            }

        return results

    def get_vectors(self, batch, vectors_type="prefix"):
        args = self.args

        if vectors_type == "prefix" or vectors_type == "both":
            prefix_encoder = self.prefix_encoder
            prefix_encoder.eval()
            prefices = batch["prefices"].to(args.device)
            prefix_masks = batch["prefix_masks"].to(args.device)
            _, _, prefix_seq_length = prefices.shape

            with torch.inference_mode():
                prefix_outs = prefix_encoder(
                    input_ids=prefices.reshape(-1, prefix_seq_length),
                    attention_mask=prefix_masks.reshape(-1, prefix_seq_length)
                )
                prefix_outs = prefix_outs.pooler_output
        if vectors_type == "suffix" or vectors_type == "both":
            suffix_encoder = self.suffix_encoder
            suffix_encoder.eval()
            suffices = batch["suffices"].to(args.device)
            suffix_masks = batch["suffix_masks"].to(args.device)
            _, _, suffix_seq_length = suffices.shape

            with torch.inference_mode():
                suffix_outs = suffix_encoder(
                    input_ids=suffices.reshape(-1, suffix_seq_length),
                    attention_mask=suffix_masks.reshape(-1, suffix_seq_length)
                )
                suffix_outs = suffix_outs.pooler_output

        if vectors_type == "prefix":
            return prefix_outs
        elif vectors_type == "suffix":
            return suffix_outs
        else:
            return prefix_outs, suffix_outs
