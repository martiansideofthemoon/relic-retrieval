import argparse
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="RELiC", type=str)
parser.add_argument('--split', default="test_with_key", type=str)
parser.add_argument('--submission', default="retriever_train/saved_models/model_0/test_submission.json", type=str)
args = parser.parse_args()

with open(f"{args.input_dir}/{args.split}.json", "r") as f:
    data = json.loads(f.read())

with open(args.submission, "r") as f:
    submission = json.loads(f.read())

NUM_SENTS = 6
results = {
    ns: {
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "recall@50": [],
        "recall@100": []
    }
    for ns in range(1, NUM_SENTS)
}

for book_title, book_data in data.items():
    for quote_id, quote_data in book_data["quotes"].items():
        submission_ranks = submission[quote_id]
        gold_answer = quote_data[1]
        for k in [1, 3, 5, 10, 50, 100]:
            results[quote_data[2]][f"recall@{k}"].append(gold_answer in submission_ranks[:k])

for ns in range(1, NUM_SENTS):
    print(f"\nResults with {ns} sentence quotes ({len(results[ns]['recall@1'])} instances):")
    for key in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100"]:
        print(
            f"{key} = {np.mean(results[ns][key]):.4f}", end=', '
        )
    print("")

# print overall results
print(f"\nResults with all quotes ({sum([len(results[ns]['recall@1']) for ns in range(1, NUM_SENTS)])} instances):")
for key in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100"]:
    all_results = [x for ns in range(1, NUM_SENTS) for x in results[ns][key]]
    print(
        f"{key} = {np.mean(all_results):.4f}", end=', '
    )
print("")
