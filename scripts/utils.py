import pickle

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
