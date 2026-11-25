import functools
import torch
from training.checkpoint import load_checkpoint
from model.config import DEV_MOE
from model.generate import generate
from model.transformer import Transformer
from tokenizer.tokenizer import encode, decode, build_vocab, load_tokenizer
from pathlib import Path

TOKENIZER_PATH = Path("artifacts/tokenizer/fr_code_32k.json")
MAX_NEW_TOKENS = 100

@functools.cache
def load_model():
    data = load_checkpoint("latest.pt")
    model = Transformer(DEV_MOE)
    model.load_state_dict(data["model"])
    model.to("cpu")
    return model


if __name__ == "__main__":

    merges, pattern, specials = load_tokenizer(TOKENIZER_PATH)
    vocab = build_vocab(merges, specials)

    model = load_model()

    question = input("\nQuelle est ta question ?\n\n").split()
    question = " ".join(question)
    idx = torch.tensor(encode(question, merges, pattern)).view(1,-1)
    réponse = generate(model, idx, MAX_NEW_TOKENS, temperature = 0.2)
    print(f"\n {decode(réponse[0].tolist(), vocab)[len(question):]}")