import argparse
import json
import numpy as np
import tqdm
import os
import re
import torch

from retriever_train.inference_utils import PrefixSuffixWrapper

from utils import build_lit_instance, build_candidates, print_results, NUM_SENTS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--model', default="retriever_train/saved_models/model_0", type=str)
parser.add_argument('--output_dir', default=None, type=str)
args = parser.parse_args()

with open(args.input_file) as f:
    data = json.loads(f.read())

retriever = PrefixSuffixWrapper(args.model, config_only=False)

left_right_re = re.compile(
    r"left_(\d+)_right_(\d+)_"
)
left_sents, right_sents = left_right_re.findall(retriever.args.data_dir)[0]
left_sents, right_sents = int(left_sents), int(right_sents)

print(f"Retriever {args.model} loaded which has {left_sents} left, {right_sents} right sentences")

BATCH_SIZE = 100

results = {
    ns: {
        "mean_rank": [],
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "recall@50": [],
        "recall@100": [],
        "num_candidates": []
    }
    for ns in range(1, NUM_SENTS)
}
total = 0
submission_data = {}


for book_title, book_data in data.items():
    # quick evaluation of only the first book

    all_quotes = [{"id": k, "quote": v} for k, v in book_data["quotes"].items()]
    all_sents = book_data["sentences"]
    # First, encode all the candidates which will be passed through suffix encoder
    candidates = build_candidates(book_data)

    all_suffixes = {}
    for ns, cands in tqdm.tqdm(candidates.items(), desc=f"Encoding suffixes for {book_title}...", total=NUM_SENTS - 1):
        all_suffixes[ns] = []
        # minibatching it to save GPU memory
        for inst_num in range(0, len(cands), BATCH_SIZE):
            # The API accepts raw strings and does the tokenization for you
            all_suffixes[ns].append(
                retriever.encode_batch(cands[inst_num:inst_num + BATCH_SIZE], vectors_type="suffix")
            )
        all_suffixes[ns] = torch.cat(all_suffixes[ns], dim=0)

    # preprocess all literary analysis quotes to have left_sents sentences before <mask> and right_sents sentences after
    all_masked_quotes = build_lit_instance([x["quote"] for x in all_quotes], left_sents, right_sents, all_sents, append_quote=False)
    # break up data by sentence length
    all_quotes_len_dict = {k: [] for k in range(1, NUM_SENTS)}
    for aq, amq in zip(all_quotes, all_masked_quotes):
        all_quotes_len_dict[aq["quote"][2]].append([aq, amq])

    # encode the prefixes using the prefix encoder
    all_prefices = {}
    for ns, qts in tqdm.tqdm(all_quotes_len_dict.items(), desc=f"Encoding prefixes for {book_title}...", total=NUM_SENTS - 1):
        all_prefices[ns] = []
        # minibatching it to save GPU memory
        for inst_num in range(0, len(qts), BATCH_SIZE):
            all_prefices[ns].append(
                retriever.encode_batch([x[1] for x in qts[inst_num:inst_num + BATCH_SIZE]], vectors_type="prefix")
            )
        if len(all_prefices[ns]) > 0:
            # This condition is necessary since an empty list gives an error for torch.cat(...)
            all_prefices[ns] = torch.cat(all_prefices[ns], dim=0)

    for ns in range(1, NUM_SENTS):
        if len(all_prefices[ns]) == 0:
            # if no quotes for this length in this book, continue
            continue

        # compute inner product between all pairs of quotes and candidates of same number of sentences
        # also compute ranks of candidates using argsort
        with torch.inference_mode():
            similarities = torch.matmul(all_prefices[ns], all_suffixes[ns].t())
            sorted_scores = torch.sort(similarities, dim=1, descending=True)
            sorted_score_idx, sorted_score_vals = sorted_scores.indices, sorted_scores.values

        ranks = []
        for qnum, (quote_data, context) in enumerate(all_quotes_len_dict[ns]):
            quote_id, quote = quote_data["id"], quote_data["quote"]
            assert ns == quote[2]
            # map the gold quote to the position in candidate list
            gold_answer = quote[1]

            if gold_answer is None:
                # this is the hidden test set, output quote_id ---> top 100 ranks list
                top_100_idx = sorted_score_idx[qnum].cpu().tolist()[:100]
                submission_data[quote_id] = top_100_idx
                continue

            # get final rank by looking up rank list

            else:
                gold_rank = sorted_score_idx[qnum].tolist().index(gold_answer) + 1
                ranks.append(gold_rank)
                '''if args.cache_scores:
                    quote.extend([gold_rank, sorted_score_idx[qnum].cpu().tolist(), sorted_score_vals[qnum].cpu().tolist()])
                else:'''
                quote.extend([gold_rank, sorted_score_idx[qnum].cpu().tolist(), None])

        results[ns]["mean_rank"].extend(ranks)
        results[ns]["recall@1"].extend([x <= 1 for x in ranks])
        results[ns]["recall@3"].extend([x <= 3 for x in ranks])
        results[ns]["recall@5"].extend([x <= 5 for x in ranks])
        results[ns]["recall@10"].extend([x <= 10 for x in ranks])
        results[ns]["recall@50"].extend([x <= 50 for x in ranks])
        results[ns]["recall@100"].extend([x <= 100 for x in ranks])
        num_cands = len(book_data["candidates"][f"{ns}_sentence"])
        results[ns]["num_candidates"].extend([num_cands for _ in ranks])

print(results)
print(submission_data)

#print_results(results)

print("Single evaluation Data")
with open(f"{args.output_dir}/single_evaluation_results.json", "w") as f:
    f.write(json.dumps(submission_data))

'''

if not load_existing and (args.cache or args.rewrite_cache) and not args.eval_small and len(submission_data) == 0:
    with open(f"{args.output_dir}/{args.split}_with_ranks.json", "w") as f:
        f.write(json.dumps(data))

if len(submission_data) > 0 and not args.eval_small:
    with open(f"{args.output_dir}/{args.split}_submission.json", "w") as f:
        f.write(json.dumps(submission_data))
    print(f"Output ranks to {args.output_dir}/{args.split}_submission.json")

'''
