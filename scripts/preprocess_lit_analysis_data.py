import argparse
import json
import numpy as np
import os
import random
import tqdm

from transformers import RobertaTokenizerFast
from utils import pickle_dump, build_lit_instance

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="/mnt/nfs/work1/miyyer/datasets/relic", type=str)
parser.add_argument('--split', default="train", type=str)
parser.add_argument('--output_dir', default="relic_preprocessed", type=str)
parser.add_argument('--total', default=1, type=int)
parser.add_argument('--augmentations', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--left_sents', default=4, type=int)
parser.add_argument('--right_sents', default=4, type=int)
parser.add_argument('--negative_examples', default=100, type=int)
args = parser.parse_args()

# mean sentence length in lit analysis is 31 tokens by roberta tokenizer
# mean quote length from lit is 67 tokens by roberta tokenizer

with open(f"{args.input_dir}/{args.split}.json", "r") as f:
    data = json.loads(f.read())

dataset = []
book_names = [k for k in data.keys()]
sent_lens = []
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

for book_title, book_data in data.items():
    all_sents = book_data[1]["sentences"]
    all_quotes = [v for k, v in book_data[0]["quotes"].items()]
    for qk, quote in book_data[0]["quotes"].items():
        quote.append(" ".join(all_sents[quote[2]:quote[2] + quote[3]]))

for ag in range(args.augmentations):
    if args.split != "train" and ag > 0:
        break
    for book_title, book_data in data.items():
        all_quotes = [v for k, v in book_data[0]["quotes"].items()]
        # hack to augment dataset with same quotes but different negative samples
        if ag > 0:
            random.shuffle(all_quotes)
        # if not enough negative samples, append some from the an alternative book to form minibatch
        if len(all_quotes) < args.negative_examples:
            alt_book = book_title
            while alt_book == book_title:
                alt_book = random.choice(book_names)
            alt_book_quotes = [v for k, v in data[alt_book][0]["quotes"].items()]
            extra_quotes = random.sample(alt_book_quotes, k=args.negative_examples - len(all_quotes))
            batch_quotes = all_quotes + extra_quotes
            dataset.append(
                build_lit_instance(batch_quotes, args.left_sents, args.right_sents)
            )
            continue

        for i in range(0, len(all_quotes), args.negative_examples):
            batch_quotes = all_quotes[i:i + args.negative_examples]
            if len(batch_quotes) < args.negative_examples:
                # append some extra quotes from same book
                extra_quotes = random.sample(all_quotes[:i], k=args.negative_examples - len(batch_quotes))
                batch_quotes = batch_quotes + extra_quotes
            dataset.append(
                build_lit_instance(batch_quotes, args.left_sents, args.right_sents)
            )

output_path = f"{args.output_dir}/left_{args.left_sents}_right_{args.right_sents}_neg_{args.negative_examples}"

if args.augmentations > 1:
    output_path += f"_aug_{args.augmentations}"

os.makedirs(output_path, exist_ok=True)
avg_context_length = np.mean([len(x[0].split()) for batch in dataset for x in batch])
avg_quote_length = np.mean([len(x[1].split()) for batch in dataset for x in batch])
print(f"Total instances = {len(dataset)}, avg context length = {avg_context_length} words, avg quote length = {avg_quote_length} words")

if args.split == "val":
    args.split = "dev"

pickle_dump(f"{output_path}/{args.split}.pkl", dataset)
