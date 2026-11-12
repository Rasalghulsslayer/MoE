"""Banc d'essai : vitesse de génération avec et sans KV cache, selon la longueur générée."""
import time

import torch

from model.config import DEV
from model.generate import generate
from model.transformer import Transformer
from utils.runtime import get_device, set_seed

PROMPT_LEN = 16
LENGTHS = [50, 100, 200, 400]
REPEATS = 3


def sync(device: torch.device) -> None:
    """Attend la fin des calculs en cours sur le device, pour un chronométrage juste."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_generation(model: Transformer, prompt: torch.Tensor, n_new: int,
                    use_cache: bool, device: torch.device) -> float:
    """Renvoie le meilleur temps, en secondes, sur REPEATS générations greedy."""
    best = float("inf")
    for _ in range(REPEATS):
        sync(device)
        t0 = time.perf_counter()
        generate(model, prompt, n_new, temperature=0.0, use_cache=use_cache)
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    set_seed(0)
    device = get_device()
    model = Transformer(DEV).to(device)
    prompt = torch.randint(0, DEV.vocab_size, (1, PROMPT_LEN), device=device)
    assert PROMPT_LEN + max(LENGTHS) <= DEV.context_length, "Contexte trop court pour ces longueurs"

    # Échauffement : les premiers appels sur MPS incluent des coûts d'initialisation
    generate(model, prompt, 10, temperature=0.0, use_cache=True)
    generate(model, prompt, 10, temperature=0.0, use_cache=False)

    print(f"Device : {device} | config DEV | prompt de {PROMPT_LEN} tokens | meilleur de {REPEATS} essais")
    print(f"{'tokens':>8} {'sans cache':>14} {'avec cache':>14} {'gain':>8}")
    for n_new in LENGTHS:
        t_without = time_generation(model, prompt, n_new, False, device)
        t_with = time_generation(model, prompt, n_new, True, device)
        print(f"{n_new:>8} {n_new / t_without:>10.0f} t/s {n_new / t_with:>10.0f} t/s "
              f"{t_without / t_with:>7.1f}×")  