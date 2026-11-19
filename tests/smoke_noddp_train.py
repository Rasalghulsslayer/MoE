"""Mini-entraînement distribué : 2 processus, 4 steps, config TEST, k=1."""
from model.config import TEST
import numpy as np
import torch
from pathlib import Path
from utils.runtime import get_device, set_seed
from training.loop import train_step
from training.dataset import load_tokens, make_sequences, get_batch_from_indices
from model.transformer import Transformer

if __name__ == "__main__":
    set_seed(42)

    cfg = TEST

    model = Transformer(cfg)
    model.to(get_device("cpu"))


    tokens = load_tokens(f"data/tokens/train_fr.bin")
    sequences = make_sequences(tokens, cfg.context_length)
    x, y = get_batch_from_indices(sequences, np.arange(0,16))

    loss = train_step(model, x, y)

    Path("tests/smoke_results").mkdir(parents=True, exist_ok=True)
    torch.save({"loss": loss, "grads": [p.grad for p in model.parameters()]}, "tests/smoke_results/not_dist_grads.pt")