import numpy as np
import torch
from pathlib import Path

TOKENS_DIR = Path("data/tokens")

def load_tokens(path : Path | str) -> np.ndarray:
    """Permet de d'accéder aux tokens sans les charger en mémoire"""
    return np.memmap(path, dtype='uint16')

def make_sequences(tokens: np.ndarray, T : int) -> np.ndarray:
    """Transforme la suite de tokens en séquences précise de T+1 éléments"""
    n = len(tokens) // (T+1)
    sequences = tokens[:n * (T+1)].reshape(n, T+1)
    return sequences

def get_batch(sequences, B, rng: np.random.Generator):
    """Extrait des séquences aléatoires du corpus pour construire les batchs."""
    idx = rng.choice(len(sequences), B, replace=False) # Sans remise
    x, y = sequences[idx, :-1], sequences[idx, 1:]
    x, y = torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)
    return x, y




if __name__ == "__main__":

    file = "train_fr"

    train_fr_tokens = load_tokens(TOKENS_DIR / f"{file}.bin")

    print(f"Nombre de token dans {file} : {len(train_fr_tokens)}")
    print(f"Nombre d'octets dans {file} : {(TOKENS_DIR / f'{file}.bin').stat().st_size}")
    print(f"20 premiers tokens de {file} : {train_fr_tokens[:20]}")