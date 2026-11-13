from model.config import ModelConfig
from model.transformer import Transformer
from training.dataset import get_batch, make_sequences, load_tokens
from utils.runtime import set_seed, get_device
import numpy as np
import torch
import time

def train_step(model: Transformer, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """Un unique pas d'entraînement"""
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train(cfg : ModelConfig, batch_size: int, n_steps : int, corpus: str, device_pref=None):
    """Boucle d'entrainement complète"""

    set_seed(42)
    rng = np.random.default_rng(42)

    device = get_device(device_pref)
    model = Transformer(cfg) 
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    tokens = load_tokens(f"data/tokens/{corpus}.bin")
    sequences = make_sequences(tokens, cfg.context_length)

    t_1 = time.perf_counter()
    train_loss = []

    for i in range (n_steps):

        t_step = time.perf_counter()

        x, y = get_batch(sequences, batch_size, rng)

        x = x.to(device)
        y = y.to(device)

        train_loss.append(train_step(model, optimizer, x, y))

        if i % 10 == 0:
            t_2 = time.perf_counter()
            print(f"Modèle en cours d'entrainement, step: {i} / {n_steps}")
            print(f"Perte actuelle : {train_loss[-1]}")
            print(f"Temps d'entrainement du dernier step : {t_2 - t_step} secondes")
            print(f"Temps d'entrainement total : {t_2 - t_1} secondes")

    return train_loss

