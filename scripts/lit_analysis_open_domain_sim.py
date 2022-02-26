import argparse
import glob
import json
import numpy as np
import tqdm
import os
import random
import re
import spacy
import torch

from retriever_train.inference_utils import PrefixSuffixWrapper
from utils import execute_gpt2, cudafy_tokens, export_server, clean_token, pickle_dump, pickle_load, form_partitions, expand_prefix, build_lit_instance
from similarity.test_sim import encode_text
from similarity.test_sim import model as sim_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"

nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="/mnt/nfs/work1/miyyer/datasets/relic", type=str)
parser.add_argument('--split', default="test", type=str)
parser.add_argument('--model', default="retriever_train/saved_models/model_49", type=str)
parser.add_argument('--total', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--left_sents', default=4, type=int)
parser.add_argument('--right_sents', default=4, type=int)
parser.add_argument('--output_dir', default=None, type=str)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.model

if os.path.exists(f"{args.model}/{args.split}_with_ranks_sim_left_{args.left_sents}_right_{args.right_sents}.json"):
    load_existing = True
    with open(f"{args.model}/{args.split}_with_ranks_sim_left_{args.left_sents}_right_{args.right_sents}.json", "r") as f:
        data = json.loads(f.read())
else:
    load_existing = False
    with open(f"{args.input_dir}/{args.split}.json", "r") as f:
        data = json.loads(f.read())


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
    all_quotes = [v for k, v in book_data[0]["quotes"].items()]
    all_sentences = book_data[1]["sentences"]

    # First, encode all the candidates which will be passed through suffix encoder
    candidates = {}
    for ns in range(1, NUM_SENTS):
        candidates[ns] = [" ".join(all_sentences[idx:idx + ns]) for idx in book_data[2]["candidates"][f"{ns}_sentence"]]

    all_suffixes = {}
    for ns, cands in tqdm.tqdm(candidates.items(), desc="Encoding suffixes...", total=NUM_SENTS - 1):
        all_suffixes[ns] = []
        # ignore encoding if ranks already available
        if load_existing:
            continue
        # minibatching it to save GPU memory
        for inst_num in range(0, len(cands), BATCH_SIZE):
            # The API accepts raw strings and does the tokenization for you
            with torch.no_grad():
                sim_tensors = encode_text(cands[inst_num:inst_num + BATCH_SIZE])
                all_suffixes[ns].append(sim_tensors)
        all_suffixes[ns] = torch.cat(all_suffixes[ns], dim=0)

    # preprocess all literary analysis quotes to have left_sents sentences before <mask> and right_sents sentences after
    all_masked_quotes = build_lit_instance(all_quotes, args.left_sents, args.right_sents, append_quote=False)
    # break up data by sentence length
    all_quotes_len_dict = {k: [] for k in range(1, NUM_SENTS)}
    for aq, amq in zip(all_quotes, all_masked_quotes):
        all_quotes_len_dict[aq[3]].append([aq, amq])

    # encode the prefixes using the prefix encoder
    all_prefices = {}
    for ns, qts in tqdm.tqdm(all_quotes_len_dict.items(), desc="Encoding prefixes...", total=NUM_SENTS - 1):
        all_prefices[ns] = []
        # ignore encoding if ranks already available
        if load_existing:
            continue
        # minibatching it to save GPU memory
        for inst_num in range(0, len(qts), BATCH_SIZE):
            with torch.no_grad():
                sim_tensors = encode_text([x[1] for x in qts[inst_num:inst_num + BATCH_SIZE]])
                all_prefices[ns].append(sim_tensors)
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
                vecs1_norm = torch.norm(all_prefices[ns], dim=1, keepdim=True)
                vecs2_norm = torch.norm(all_suffixes[ns], dim=1, keepdim=True)
                norm_product = torch.matmul(vecs1_norm, vecs2_norm.t())
                assert norm_product.shape == similarities.shape
                similarities = torch.div(similarities, norm_product)
                sorted_scores = torch.argsort(similarities, dim=1, descending=True)

        ranks = []
        for qnum, (quote, context) in enumerate(all_quotes_len_dict[ns]):
            assert ns == quote[3]
            # map the gold quote to the position in candidate list
            gold_answer = quote[2]
            try:
                gold_candidate_index = book_data[2]["candidates"][f"{ns}_sentence"].index(gold_answer)
            except:
                import pdb; pdb.set_trace()
                pass
            # get final rank by looking up rank list
            if load_existing:
                ranks.append(quote[-3])
            else:
                gold_rank = sorted_scores[qnum].tolist().index(gold_candidate_index) + 1
                ranks.append(gold_rank)
                quote.extend([gold_rank, gold_candidate_index, sorted_scores[qnum].cpu().tolist()])

        results[ns]["mean_rank"].extend(ranks)
        results[ns]["recall@1"].extend([x <= 1 for x in ranks])
        results[ns]["recall@3"].extend([x <= 3 for x in ranks])
        results[ns]["recall@5"].extend([x <= 5 for x in ranks])
        results[ns]["recall@10"].extend([x <= 10 for x in ranks])
        results[ns]["recall@50"].extend([x <= 50 for x in ranks])
        results[ns]["recall@100"].extend([x <= 100 for x in ranks])
        results[ns]["recall@1650"].extend([x <= 1650 for x in ranks])
        num_cands = len(book_data[2]["candidates"][f"{ns}_sentence"])
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

if not load_existing:
    with open(f"{args.output_dir}/{args.split}_with_ranks_sim_left_{args.left_sents}_right_{args.right_sents}.json", "w") as f:
        f.write(json.dumps(data))
