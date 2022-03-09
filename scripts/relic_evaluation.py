import argparse
import json
import numpy as np
import tqdm
import os
import re
import torch

from retriever_train.inference_utils import PrefixSuffixWrapper
from utils import build_lit_instance

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="RELiC", type=str)
parser.add_argument('--split', default="test", type=str)
parser.add_argument('--model', default="retriever_train/saved_models/model_0", type=str)
parser.add_argument('--total', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--output_dir', default=None, type=str)
parser.add_argument('--write_to_file', action='store_true')
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.model

if os.path.exists(f"{args.model}/{args.split}_with_ranks.json"):
    load_existing = True
    with open(f"{args.model}/{args.split}_with_ranks.json", "r") as f:
        data = json.loads(f.read())
else:
    load_existing = False
    with open(f"{args.input_dir}/{args.split}.json", "r") as f:
        data = json.loads(f.read())

retriever = PrefixSuffixWrapper(args.model, config_only=load_existing)

left_right_re = re.compile(
    r"left_(\d+)_right_(\d+)_"
)
left_sents, right_sents = left_right_re.findall(retriever.args.data_dir)[0]
left_sents, right_sents = int(left_sents), int(right_sents)

print(f"Retriever {args.model} loaded which has {left_sents} left, {right_sents} right sentences")

BATCH_SIZE = 100

NUM_SENTS = 6
results = {
    ns: {
        "mean_rank": [],
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "recall@50": [],
        "recall@100": [],
        "recall@1650": [],
        "num_candidates": []
    }
    for ns in range(1, NUM_SENTS)
}
total = 0

for book_title, book_data in data.items():
    all_quotes = [v for k, v in book_data["quotes"].items()]
    all_sentences = book_data["sentences"]

    # First, encode all the candidates which will be passed through suffix encoder
    candidates = {}
    for ns in range(1, NUM_SENTS):
        candidates[ns] = [" ".join([x.strip() for x in all_sentences[idx:idx + ns]]) for idx in book_data["candidates"][f"{ns}_sentence"]]

    all_suffixes = {}
    for ns, cands in tqdm.tqdm(candidates.items(), desc="Encoding suffixes...", total=NUM_SENTS - 1):
        all_suffixes[ns] = []
        # ignore encoding if ranks already available
        if load_existing:
            continue
        # minibatching it to save GPU memory
        for inst_num in range(0, len(cands), BATCH_SIZE):
            # The API accepts raw strings and does the tokenization for you
            all_suffixes[ns].append(
                retriever.encode_batch(cands[inst_num:inst_num + BATCH_SIZE], vectors_type="suffix")
            )
        all_suffixes[ns] = torch.cat(all_suffixes[ns], dim=0)

    # preprocess all literary analysis quotes to have left_sents sentences before <mask> and right_sents sentences after
    all_masked_quotes = build_lit_instance(all_quotes, left_sents, right_sents, append_quote=False)
    # break up data by sentence length
    all_quotes_len_dict = {k: [] for k in range(1, NUM_SENTS)}
    for aq, amq in zip(all_quotes, all_masked_quotes):
        all_quotes_len_dict[aq[2]].append([aq, amq])

    # encode the prefixes using the prefix encoder
    all_prefices = {}
    for ns, qts in tqdm.tqdm(all_quotes_len_dict.items(), desc="Encoding prefixes...", total=NUM_SENTS - 1):
        all_prefices[ns] = []
        # ignore encoding if ranks already available
        if load_existing:
            continue
        # minibatching it to save GPU memory
        for inst_num in range(0, len(qts), BATCH_SIZE):
            all_prefices[ns].append(
                retriever.encode_batch([x[1] for x in qts[inst_num:inst_num + BATCH_SIZE]], vectors_type="prefix")
            )
        if len(all_prefices[ns]) > 0:
            # This condition is necessary since an empty list gives an error for torch.cat(...)
            all_prefices[ns] = torch.cat(all_prefices[ns], dim=0)

    for ns in range(1, NUM_SENTS):
        if not load_existing and len(all_prefices[ns]) == 0:
            # if no quotes for this length in this book, print results for this length and move on to next length/book
            print(f"\nResults with {ns} sentence quotes ({len(results[ns]['mean_rank'])} instances):")
            for key in ["mean_rank", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "num_candidates"]:
                print(
                    f"{key} = {np.mean(results[ns][key]):.4f}", end=', '
                )
            print("")
            continue

        # compute inner product between all pairs of quotes and candidates of same number of sentences
        # also compute ranks of candidates using argsort
        if not load_existing:
            with torch.no_grad():
                similarities = torch.matmul(all_prefices[ns], all_suffixes[ns].t())
                sorted_scores = torch.sort(similarities, dim=1, descending=True)
                sorted_score_idx, sorted_score_vals = sorted_scores.indices, sorted_scores.values

        ranks = []
        for qnum, (quote, context) in enumerate(all_quotes_len_dict[ns]):
            assert ns == quote[2]
            # map the gold quote to the position in candidate list
            gold_answer = quote[1]
            try:
                gold_candidate_index = book_data["candidates"][f"{ns}_sentence"].index(gold_answer)
            except:
                import pdb; pdb.set_trace()
                pass
            # get final rank by looking up rank list
            if load_existing:
                ranks.append(quote[-4])
            else:
                gold_rank = sorted_score_idx[qnum].tolist().index(gold_candidate_index) + 1
                ranks.append(gold_rank)
                quote.extend([gold_rank, gold_candidate_index, sorted_score_idx[qnum].cpu().tolist(), sorted_score_vals[qnum].cpu().tolist()])

        results[ns]["mean_rank"].extend(ranks)
        results[ns]["recall@1"].extend([x <= 1 for x in ranks])
        results[ns]["recall@3"].extend([x <= 3 for x in ranks])
        results[ns]["recall@5"].extend([x <= 5 for x in ranks])
        results[ns]["recall@10"].extend([x <= 10 for x in ranks])
        results[ns]["recall@50"].extend([x <= 50 for x in ranks])
        results[ns]["recall@100"].extend([x <= 100 for x in ranks])
        results[ns]["recall@1650"].extend([x <= 1650 for x in ranks])
        num_cands = len(book_data["candidates"][f"{ns}_sentence"])
        results[ns]["num_candidates"].extend([num_cands for _ in ranks])

        print(f"\nResults with {ns} sentence quotes ({len(results[ns]['mean_rank'])} instances):")
        for key in ["mean_rank", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100", "recall@1650", "num_candidates"]:
            print(
                f"{key} = {np.mean(results[ns][key]):.4f}", end=', '
            )
        print("")

    # print overall results
    print(f"\nResults with all quotes ({sum([len(results[ns]['mean_rank']) for ns in range(1, NUM_SENTS)])} instances):")
    for key in ["mean_rank", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100", "recall@1650", "num_candidates"]:
        all_results = [x for ns in range(1, NUM_SENTS) for x in results[ns][key]]
        print(
            f"{key} = {np.mean(all_results):.4f}", end=', '
        )
    print("")

if not load_existing and args.write_to_file:
    with open(f"{args.output_dir}/{args.split}_with_ranks.json", "w") as f:
        f.write(json.dumps(data))
