"""
python scripts/score_submission.py --key_file RELiC/test_key.json --submission retriever_train/saved_models/model_0/test_submission.json
"""

import argparse
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--key_file', default="RELiC/test_key.json", type=str)
parser.add_argument('--submission', default="retriever_train/saved_models/model_0/test_submission.json", type=str)
args = parser.parse_args()

with open(args.key_file, "r") as f:
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

        # checking if the quality of the submitted data is okay
        assert isinstance(submission_ranks, list)
        assert len(submission_ranks) == 100
        assert all(isinstance(x, int) for x in submission_ranks)

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
total_instances = sum([len(results[ns]['recall@1']) for ns in range(1, NUM_SENTS)])
print(f"\nResults with all quotes ({total_instances} instances):")
for key in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100"]:
    all_results = [x for ns in range(1, NUM_SENTS) for x in results[ns][key]]
    print(
        f"{key} = {np.mean(all_results):.4f}", end=', '
    )
print("")
