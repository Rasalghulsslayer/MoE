import numpy as np
import torch
from pathlib import Path

TOKENS_DIR = Path("data/tokens")

def load_tokens(path : Path) -> np.ndarray:
    """Charges les tokens en mémoire"""
    return np.memmap(path, dtype='uint16')

def make_sequences(tokens: np.ndarray, T : int) -> np.ndarray:
    """Transforme la suite de tokens en séquences précise de T+1 éléments"""
    sequences = []
    i = 0
    while ((i + T + 1) < len(tokens)):
        sequences.append(tokens[i:i+T+1])
        i += T+1
    return np.array(sequences)

def get_batch(sequences, B, rng: np.random.default_rng):
    idx = rng.choice(len(sequences), B)
    x, y = sequences[idx, :-1], sequences[idx, 1:]
    x, y = torch.tensor(x), torch.tensor(y)
    return x, y




if __name__ == "__main__":

    file = "train_fr"

    train_fr_tokens = load_tokens(TOKENS_DIR / f"{file}.bin")

    print(f"Nombre de token dans {file} : {len(train_fr_tokens)}")
    print(f"Nombre d'octets dans {file} : {(TOKENS_DIR / f'{file}.bin').stat().st_size}")
    print(f"20 premiers tokens de {file} : {train_fr_tokens[:20]}")