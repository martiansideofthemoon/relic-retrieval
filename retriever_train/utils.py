import logging
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import namedtuple

from functools import partial

from dataset_config import DATASET_CONFIG
from data_utils import update_config, Instance


logger = logging.getLogger(__name__)


def class_number_to_str(eval_dataset, class_number):
    if isinstance(class_number, str):
        return ", ".join(["{} {}".format(x.split("-")[0], x.split("-")[1]) for x in class_number.split("_")])
    else:
        return eval_dataset.reverse_label_dict[class_number.item()]

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
            batch_size, num_pairs, prefix_seq_length = prefices.shape

            with torch.no_grad():
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
            batch_size, num_pairs, suffix_seq_length = suffices.shape

            with torch.no_grad():
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


def get_logits(model, iteration, generated, segments, style_content_vectors, past):
    if iteration == 0:
        if style_content_vectors is None:
            pred = model(
                input_ids=generated,
                token_type_ids=segments
            )
        else:
            pred = model(
                input_ids=generated,
                token_type_ids=segments,
                prefix_input_vectors=style_content_vectors
            )
    else:
        # used the cached representations to speed up decoding
        pred = model(
            input_ids=generated[:, -1:],
            token_type_ids=segments[:, -1:],
            past_key_values=past
        )
    logits = pred['logits']
    past = pred['past_key_values']
    return logits, past


def sample_sequence(model, length, context, style_content_vectors, segments, eos_token_id,
                    num_samples=1, temperature=1, top_k=0, top_p=0.0, get_scores=False,
                    interpolation=None):

    if length is None and style_content_vectors is not None:
        new_length = 1024 - style_content_vectors.shape[1] - context.shape[1]
    elif length is None and style_content_vectors is None:
        new_length = 1024 - context.shape[1]
    else:
        new_length = length

    batch_size = context.shape[0]

    eos_emitted = [False for _ in range(batch_size)]

    generated = context
    scores = [{"score": 0, "sequence": []} for _ in range(batch_size)]

    with torch.no_grad():
        past = None
        past2 = None
        for i in range(new_length):
            logits, past = get_logits(
                model, i, generated, segments, style_content_vectors, past
            )
            if interpolation:
                logits2, past2 = get_logits(
                    model=interpolation["model"].roberta_gpt2.gpt2,
                    iteration=i,
                    generated=generated,
                    segments=segments,
                    style_content_vectors=style_content_vectors,
                    past=past2
                )
                probs = F.softmax(logits[:, -1, :], dim=-1)
                probs2 = F.softmax(logits2[:, -1, :], dim=-1)
                final_probs = interpolation["weight"] * probs2 + (1 - interpolation["weight"]) * probs
                next_token_logits = torch.log(final_probs) / (temperature if temperature > 0 else 1.)
                original_scores = next_token_logits.clone()
            else:
                next_token_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.)
                original_scores = F.log_softmax(next_token_logits, dim=-1)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0 and top_k in [0, 1] and top_p == 0.0:
                # greedy sampling
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            if get_scores:
                for batch_elem in range(batch_size):
                    if eos_emitted[batch_elem]:
                        continue
                    scores[batch_elem]["score"] += original_scores[batch_elem, next_token[batch_elem].item()].item()
                    scores[batch_elem]["sequence"].append("token")

            generated = torch.cat((generated, next_token), dim=1)
            segments = torch.cat((segments, segments[:, -1:]), dim=1)

            for batch_elem in range(batch_size):
                if next_token[batch_elem].item() == eos_token_id:
                    eos_emitted[batch_elem] = True

            if length is None and all(eos_emitted):
                break

    if get_scores:
        scores = [score_fn(x, True) for x in scores]

    return generated, scores


def score_fn(x, length_normalize):
    if length_normalize:
        return x["score"] / len(x["sequence"])
    else:
        return x["score"]


def beam_search(model, length, context, style_content_vectors, segments, eos_token_id,
                beam_size=1, beam_search_scoring="normalize"):

    def merge_pasts(all_beams, prev_past):
        past_indices = [beam["past"] for element in all_beams for beam in element]
        return [pp[:, past_indices, :, :, :] for pp in prev_past]

    def merge_input_ids(all_beams):
        input_ids = [beam["sequence"][-1] for element in all_beams for beam in element]
        return torch.cat(input_ids, dim=0)

    if beam_search_scoring == "normalize":
        _score_fn = partial(score_fn, length_normalize=True)
    else:
        _score_fn = partial(score_fn, length_normalize=False)

    if length is None and style_content_vectors is not None:
        new_length = 1024 - style_content_vectors.shape[1] - context.shape[1]
    elif length is None and style_content_vectors is None:
        new_length = 1024 - context.shape[1]
    else:
        new_length = length

    with torch.no_grad():
        if style_content_vectors is None:
            logits, past = model(
                input_ids=context,
                token_type_ids=segments
            )
        else:
            logits, past = model(
                input_ids=context,
                token_type_ids=segments,
                prefix_input_vectors=style_content_vectors
            )
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        top_scores, top_indices = torch.topk(input=log_probs, k=beam_size, dim=-1)

        all_beams = []
        for elem_num, (ts, ti) in enumerate(zip(top_scores, top_indices)):
            curr_element = []
            for bs in range(beam_size):
                curr_element.append({
                    "score": ts[bs],
                    "past": elem_num,
                    "sequence": [x.unsqueeze(0).unsqueeze(0) for x in context[elem_num]] + [ti[bs].unsqueeze(0).unsqueeze(0)],
                    "eos_emitted": False
                })
            all_beams.append(curr_element)

        # one time step here since segment IDs remain constant during generation
        tiled_segments = torch.cat([segments[:, -1:] for _ in range(beam_size)], dim=-1)

        for i in range(1, new_length):
            # check if all beams have emitted an EOS token
            all_eos = all([beam["eos_emitted"] for element in all_beams for beam in element])
            if all_eos:
                break

            latest_input_ids = merge_input_ids(all_beams)
            past = merge_pasts(all_beams, past)

            logits, past = model(
                input_ids=latest_input_ids,  # input_ids[:, -1:],
                token_type_ids=tiled_segments,
                past=past
            )
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            top_scores, top_indices = torch.topk(input=log_probs, k=beam_size, dim=-1)

            new_beams = []
            curr_element = []
            for mb_num, (ts, ti) in enumerate(zip(top_scores, top_indices)):
                current_elem_num = mb_num // beam_size
                current_elem_beam_num = mb_num % beam_size
                old_beam = all_beams[current_elem_num][current_elem_beam_num]

                if old_beam["eos_emitted"]:
                    curr_element.append(old_beam)
                else:
                    for bs in range(beam_size):
                        token = ti[bs].unsqueeze(0).unsqueeze(0)
                        curr_element.append({
                            "score": old_beam["score"] + ts[bs],
                            "past": mb_num,
                            "sequence": old_beam["sequence"] + [token],
                            "eos_emitted": token.item() == eos_token_id
                        })
                if current_elem_beam_num == beam_size - 1:
                    new_beams.append(curr_element)
                    curr_element = []

            # Sort the beams by score and keep only top scoring elements
            all_beams = []
            for elem in new_beams:
                elem.sort(key=lambda x: _score_fn(x), reverse=True)
                all_beams.append(elem[:beam_size])

        final_beams = []
        for elem in all_beams:
            elem.sort(key=lambda x: _score_fn(x), reverse=True)
            # just return the highest scoring prediction
            final_beams.append(elem[:1])

        final_input_ids = [
            torch.cat(elem[0]["sequence"], dim=1).squeeze(0) for elem in final_beams
        ]

        return final_input_ids, [_score_fn(fb[0]) for fb in final_beams]
