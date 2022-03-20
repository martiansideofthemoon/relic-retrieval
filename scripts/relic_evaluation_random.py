import argparse
import json
import numpy as np
import tqdm
import os
import re
import torch

from utils import build_lit_instance, build_candidates, NUM_SENTS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="RELiC", type=str)
parser.add_argument('--split', default="val", type=str)
parser.add_argument('--num_samples', default=100, type=int)
args = parser.parse_args()

with open(f"{args.input_dir}/{args.split}.json", "r") as f:
    data = json.loads(f.read())

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

print(f"Evaluating {args.split} split with {len(data)} books...")

for samples in range(args.num_samples):
    for book_title, book_data in data.items():
        all_quotes = [v for k, v in book_data["quotes"].items()]
        # First, encode all the candidates which will be passed through suffix encoder
        candidates = build_candidates(book_data)

        # preprocess all literary analysis quotes to have left_sents sentences before <mask> and right_sents sentences after
        all_masked_quotes = build_lit_instance(all_quotes, 1, 1, append_quote=False)
        # break up data by sentence length
        all_quotes_len_dict = {k: [] for k in range(1, NUM_SENTS)}
        for aq, amq in zip(all_quotes, all_masked_quotes):
            all_quotes_len_dict[aq[2]].append([aq, amq])

        for ns in range(1, NUM_SENTS):
            with torch.inference_mode():
                similarities = torch.rand(len(all_quotes_len_dict[ns]), len(candidates[ns]))
                sorted_scores = torch.sort(similarities, dim=1, descending=True)
                sorted_score_idx, sorted_score_vals = sorted_scores.indices, sorted_scores.values

            ranks = []
            for qnum, (quote, context) in enumerate(all_quotes_len_dict[ns]):
                assert ns == quote[2]
                # map the gold quote to the position in candidate list
                gold_answer = quote[1]
                gold_candidate_index = book_data["candidates"][f"{ns}_sentence"].index(gold_answer)
                # get final rank by looking up rank list
                gold_rank = sorted_score_idx[qnum].tolist().index(gold_candidate_index) + 1
                ranks.append(gold_rank)

            results[ns]["mean_rank"].extend(ranks)
            results[ns]["recall@1"].extend([x <= 1 for x in ranks])
            results[ns]["recall@3"].extend([x <= 3 for x in ranks])
            results[ns]["recall@5"].extend([x <= 5 for x in ranks])
            results[ns]["recall@10"].extend([x <= 10 for x in ranks])
            results[ns]["recall@50"].extend([x <= 50 for x in ranks])
            results[ns]["recall@100"].extend([x <= 100 for x in ranks])
            num_cands = len(book_data["candidates"][f"{ns}_sentence"])
            results[ns]["num_candidates"].extend([num_cands for _ in ranks])

    # print overall results
    print(f"\nResults with all quotes ({sum([len(results[ns]['mean_rank']) for ns in range(1, NUM_SENTS)])} instances):")
    for key in ["mean_rank", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100", "num_candidates"]:
        all_results = [x for ns in range(1, NUM_SENTS) for x in results[ns][key]]
        print(
            f"{key} = {np.mean(all_results):.4f}", end=', '
        )
    print("")
