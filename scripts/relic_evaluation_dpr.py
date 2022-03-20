import argparse
import json
import numpy as np
import tqdm
import os
import torch

from utils import build_lit_instance, print_results, build_candidates, NUM_SENTS

from transformers import AutoTokenizer, DPRContextEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="RELiC", type=str)
parser.add_argument('--split', default="val", type=str)
parser.add_argument('--left_sents', default=4, type=int)
parser.add_argument('--right_sents', default=4, type=int)
parser.add_argument('--output_dir', default="retriever_train/saved_models/dpr_model", type=str)
parser.add_argument('--eval_small', action='store_true')
parser.add_argument('--rewrite_cache', action='store_true')
parser.add_argument('--cache_scores', action='store_true')
parser.add_argument('--cache', action='store_true')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


if not args.rewrite_cache and os.path.exists(f"{args.output_dir}/{args.split}_with_ranks_dpr_left_{args.left_sents}_right_{args.right_sents}.json"):
    load_existing = True
    with open(f"{args.output_dir}/{args.split}_with_ranks_dpr_left_{args.left_sents}_right_{args.right_sents}.json", "r") as f:
        data = json.loads(f.read())
else:
    load_existing = False
    with open(f"{args.input_dir}/{args.split}.json", "r") as f:
        data = json.loads(f.read())

tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model.cuda()

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

print(f"Evaluating {args.split} split with {len(data)} books...")

for book_title, book_data in data.items():
    # quick evaluation of only the first book
    if args.eval_small and len(results[1]["mean_rank"]) > 0:
        break

    all_quotes = [{"id": k, "quote": v} for k, v in book_data["quotes"].items()]
    # First, encode all the candidates which will be passed through suffix encoder
    candidates = build_candidates(book_data)

    all_suffixes = {}
    for ns, cands in tqdm.tqdm(candidates.items(), desc=f"Encoding suffixes for {book_title}...", total=NUM_SENTS - 1):
        all_suffixes[ns] = []
        # ignore encoding if ranks already available
        if load_existing:
            continue
        # minibatching it to save GPU memory
        for inst_num in range(0, len(cands), BATCH_SIZE):
            # The API accepts raw strings and does the tokenization for you
            with torch.inference_mode():
                dpr_tensors = tokenizer(cands[inst_num:inst_num + BATCH_SIZE], return_tensors="pt", padding=True, truncation=True, max_length=512)
                for k, v in dpr_tensors.items():
                    dpr_tensors[k] = v.cuda()
                all_suffixes[ns].append(model(**dpr_tensors).pooler_output)
        all_suffixes[ns] = torch.cat(all_suffixes[ns], dim=0)

    # preprocess all literary analysis quotes to have left_sents sentences before <mask> and right_sents sentences after
    all_masked_quotes = build_lit_instance([x["quote"] for x in all_quotes], args.left_sents, args.right_sents, append_quote=False)
    # break up data by sentence length
    all_quotes_len_dict = {k: [] for k in range(1, NUM_SENTS)}
    for aq, amq in zip(all_quotes, all_masked_quotes):
        all_quotes_len_dict[aq["quote"][2]].append([aq, amq])

    # encode the prefixes using the prefix encoder
    all_prefices = {}
    for ns, qts in tqdm.tqdm(all_quotes_len_dict.items(), desc=f"Encoding prefixes for {book_title}...", total=NUM_SENTS - 1):
        all_prefices[ns] = []
        # ignore encoding if ranks already available
        if load_existing:
            continue
        # minibatching it to save GPU memory
        for inst_num in range(0, len(qts), BATCH_SIZE):
            with torch.inference_mode():
                dpr_tensors = tokenizer([x[1] for x in qts[inst_num:inst_num + BATCH_SIZE]], return_tensors="pt", padding=True, truncation=True, max_length=512)
                for k, v in dpr_tensors.items():
                    dpr_tensors[k] = v.cuda()
                all_prefices[ns].append(model(**dpr_tensors).pooler_output)
        if len(all_prefices[ns]) > 0:
            # This condition is necessary since an empty list gives an error for torch.cat(...)
            all_prefices[ns] = torch.cat(all_prefices[ns], dim=0)

    for ns in range(1, NUM_SENTS):
        if not load_existing and len(all_prefices[ns]) == 0:
            # if no quotes for this length in this book, continue
            continue

        # compute inner product between all pairs of quotes and candidates of same number of sentences
        # also compute ranks of candidates using argsort
        if not load_existing:
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
            if load_existing:
                ranks.append(quote[-3])
            else:
                gold_rank = sorted_score_idx[qnum].tolist().index(gold_answer) + 1
                ranks.append(gold_rank)
                if args.cache_scores:
                    quote.extend([gold_rank, sorted_score_idx[qnum].cpu().tolist(), sorted_score_vals[qnum].cpu().tolist()])
                else:
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

print_results(results)

if not load_existing and (args.cache or args.rewrite_cache) and not args.eval_small and len(submission_data) == 0:
    with open(f"{args.output_dir}/{args.split}_with_ranks_dpr_left_{args.left_sents}_right_{args.right_sents}.json", "w") as f:
        f.write(json.dumps(data))

if len(submission_data) > 0 and not args.eval_small:
    with open(f"{args.output_dir}/{args.split}_left_{args.left_sents}_right_{args.right_sents}_submission.json", "w") as f:
        f.write(json.dumps(submission_data))
    print(f"Output ranks to {args.output_dir}/{args.split}_left_{args.left_sents}_right_{args.right_sents}_submission.json")
