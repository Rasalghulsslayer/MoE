from training.dataset import load_tokens, make_sequences, get_batch
import numpy as np
from tokenizer.tokenizer import decode, build_vocab, load_tokenizer

if __name__ == "__main__":

    B = 1
    T = 64

    rng = np.random.default_rng(42)

    tokens = load_tokens("data/tokens/train_fr.bin")

    sequences = make_sequences(tokens, T)

    batch = get_batch(sequences, B, rng)

    print(f"Véritable batch d'entrainement : {batch}")

    merges, pattern, specials = load_tokenizer("artifacts/tokenizer/fr_code_32k.json")

    vocab = build_vocab(merges, specials)

    x, y = batch

    print(f"x: {decode(x.tolist()[0], vocab)}")
    print(f"y: {decode(y.tolist()[0], vocab)}")