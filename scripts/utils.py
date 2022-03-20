import numpy as np
import pickle

NUM_SENTS = 6

def build_lit_instance(quotes, left_sents, right_sents, append_quote=True, mask_token="<mask>"):
    """Make a context using left / right sentences. Use <mask> token for quote."""
    inst = []
    for quote in quotes:
        if left_sents > 0:
            left_context = " ".join(quote[0][-left_sents:])
        else:
            left_context = ""
        if right_sents > 0:
            right_context = " ".join(quote[3][:right_sents])
        else:
            right_context = ""
        full_context = left_context + f" {mask_token} " + right_context

        if append_quote:
            actual_quote = quote[-1]
            inst.append(
                [" ".join(full_context.split()), " ".join(actual_quote.split())]
            )
        else:
            inst.append(
                " ".join(full_context.split())
            )
    return inst


def pickle_load(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_dump(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def build_candidates(book_data):
    candidates = {}
    all_sentences = book_data["sentences"]
    for ns in range(1, NUM_SENTS):
        candidates[ns] = [" ".join([x.strip() for x in all_sentences[idx:idx + ns]]) for idx in book_data["candidates"][f"{ns}_sentence"]]
        assert all([ii == xx for ii, xx in enumerate(book_data["candidates"][f"{ns}_sentence"])])
    return candidates


def print_results(results):
    if len(results[1]["recall@1"]) > 0:
        for ns in range(1, NUM_SENTS):
            print(f"\nResults with {ns} sentence quotes ({len(results[ns]['mean_rank'])} instances):")
            for key in ["mean_rank", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100", "num_candidates"]:
                print(
                    f"{key} = {np.mean(results[ns][key]):.4f}", end=', '
                )
            print("")

        # print overall results
        print(f"\nResults with all quotes ({sum([len(results[ns]['mean_rank']) for ns in range(1, NUM_SENTS)])} instances):")
        for key in ["mean_rank", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100", "num_candidates"]:
            all_results = [x for ns in range(1, NUM_SENTS) for x in results[ns][key]]
            print(
                f"{key} = {np.mean(all_results):.4f}", end=', '
            )
        print("")
    else:
        print("Not printing results since the answers are hidden for this split...")
