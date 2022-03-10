import torch
from similarity.sim_models import WordAveraging
from similarity.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm

tok = TreebankWordTokenizer()

model = torch.load('scripts/similarity/sim/sim.pt')
state_dict = model['state_dict']
vocab_words = model['vocab_words']
args = model['args']
# turn off gpu
model = WordAveraging(args, vocab_words)
model.load_state_dict(state_dict, strict=True)
sp = spm.SentencePieceProcessor()
sp.Load('scripts/similarity/sim/sim.sp.30k.model')
model.eval()

def make_example(sentence, model):
    sentence = sentence.lower()
    sentence = " ".join(tok.tokenize(sentence))
    sentence = sp.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1

def find_similarity(s1, s2):
    with torch.inference_mode():
        s1 = [make_example(x, model) for x in s1]
        s2 = [make_example(x, model) for x in s2]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        wx2, wl2, wm2 = model.torchify_batch(s2)
        scores = model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        return [x.item() for x in scores]

def find_similarity_matrix(s1, s2):
    with torch.inference_mode():
        s1 = [make_example(x, model) for x in s1]
        s2 = [make_example(x, model) for x in s2]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        wx2, wl2, wm2 = model.torchify_batch(s2)
        vecs1 = model.encode(idxs=wx1, mask=wm1, lengths=wl1)
        vecs2 = model.encode(idxs=wx2, mask=wm2, lengths=wl2)
        dot_product = torch.matmul(vecs1, vecs2.t())

        vecs1_norm = torch.linalg.norm(vecs1, dim=1, keepdims=True)
        vecs2_norm = torch.linalg.norm(vecs2, dim=1, keepdims=True)
        norm_product = torch.matmul(vecs1_norm, vecs2_norm.t())
    return torch.divide(dot_product, norm_product)

def encode_text(s1):
    with torch.inference_mode():
        s1 = [make_example(x, model) for x in s1]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        vecs1 = model.encode(idxs=wx1, mask=wm1, lengths=wl1)
        return vecs1
