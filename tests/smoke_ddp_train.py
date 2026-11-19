"""Mini-entraînement distribué : 2 processus, 4 steps, config TEST, k=1."""
from model.config import TEST
import numpy as np
import torch
from pathlib import Path
from utils.runtime import get_device, set_seed, setup_distributed, is_main_process
from training.train_config import TrainConfig
from training.loop import train_step
from training.dataset import load_tokens, make_sequences, get_batch_from_indices, sample_indices
from model.transformer import Transformer

if __name__ == "__main__":
    set_seed(42)

    dist = setup_distributed()

    cfg = TEST

    model = Transformer(cfg)
    model.to(get_device("cpu"))
    model = torch.nn.parallel.DistributedDataParallel(model)


    tokens = load_tokens(f"data/tokens/train_fr.bin")
    sequences = make_sequences(tokens, cfg.context_length)
    x, y = get_batch_from_indices(sequences, [np.arange(dist[0] * 8, (dist[0] + 1) * 8 - 1)])

    loss = train_step(model, x, y)
    
    torch.save({"loss": loss, "grads": [p.grad for p in model.parameters()]}, f"tests/smoke_results/dist_grads{dist[0]}.pt")