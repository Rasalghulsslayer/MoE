from tokenizer.tokenizer import encode, decode
from tokenizer.tokenizer import load_tokenizer
from model.config import ModelConfig, DEV
from model.transformer import Transformer
from model.generate import generate
from training.dataset import get_batch, make_sequences, load_tokens
from training.schedule import get_lr
from training.train_config import TrainConfig
from training.checkpoint import load_checkpoint, save_checkpoint
from utils.runtime import set_seed, get_device
from pathlib import Path

import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time

def train_step(model: Transformer, x: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    """Un unique pas d'entraînement"""
    _, loss = model(x, targets=y)
    loss = loss / k
    loss.backward()
    return loss.item()

def train(cfg: ModelConfig, schedule: TrainConfig, batch_size: int, n_steps: int, corpus: str, k: int=1, device_pref=None, resume_from = None):
    """Boucle d'entrainement complète"""
    assert batch_size % k == 0, "Mauvaise entrée pour batch_size et k, veuillez choisir batch_size multiple de k."
    device = get_device(device_pref)

    if resume_from is None:
        set_seed(42)

    rng = np.random.default_rng()
    model = Transformer(cfg) 
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.base_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: get_lr(step, schedule)/schedule.base_lr)
    train_loss = []

    if resume_from is not None:
        data = load_checkpoint(resume_from)

        if data["config"] != cfg:
            raise ValueError("Ce checkpoint n'est pas compatible avec la configuration")
        
        i = data["step"]
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        scheduler.load_state_dict(data["scheduler"])
        rng.bit_generator.state = data["rng_numpy"]
        torch.set_rng_state(data["rng_torch"])
        train_loss = data["train_loss"]
    else: 
        rng = np.random.default_rng(42)
        i=0
        

    tokens = load_tokens(f"data/tokens/train_{corpus}.bin")
    sequences = make_sequences(tokens, cfg.context_length)
    val_tokens = load_tokens(f"data/tokens/val_{corpus}.bin")
    val_sequences = make_sequences(val_tokens, cfg.context_length)
    val_loss = []

    t_1 = time.perf_counter()

    loss=0
    
    optimizer.zero_grad(set_to_none=True)

    while (i < (n_steps // k * k)):
        i+=1

        t_step = time.perf_counter()

        x, y = get_batch(sequences, int(batch_size/k), rng)

        x = x.to(device)
        y = y.to(device)

        loss += train_step(model, x, y, k=k)

        if i % k == 0:
            train_loss.append(loss)
            loss = 0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            t_2 = time.perf_counter()
            print("\n")
            print(f"===Modèle en cours d'entrainement, step: {i} / {n_steps}===")
            print(f"Learning rate : {scheduler.get_last_lr()}")
            print(f"Perte d'entrainement actuelle : {train_loss[-1]}")
            print(f"Temps d'entrainement du dernier micro-batch : {t_2 - t_step} secondes")
            t_step = time.perf_counter()
            print(f"Temps d'entrainement total : {t_2 - t_1} secondes")
            print("\n")
            
            save_checkpoint(i, model, optimizer, scheduler, rng, train_loss, cfg, schedule, save_every=50)
            eval_every = 20
            n_batches = 10
            if i % eval_every == 0: #eval_every multiple de k
                set_seed(42)
                vl = evaluate(model, val_sequences, n_batches, batch_size // k, device)
                val_loss.append(vl)
                print(f"Perplexité val : {math.exp(vl):.0f}")

            generate_every = 100
            if i % generate_every == 0: #generate_every multiple de k
                set_seed(42)
                merges, pattern, _ = load_tokenizer(Path("artifacts/tokenizer/fr_code_32k.json"))
                idx = encode("Le cha", merges, pattern)
                print(generate(model, idx, max_new_tokens=50, temperature=0.8, top_k=5, top_p=10))


    return train_loss, val_loss



def plot_loss(losses, window=20):
    """Graph de la loss."""
    mean = [
        sum(losses[max(0, i - window + 1):i + 1]) / (i - max(0, i - window + 1) + 1)
        for i in range(len(losses))
    ]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(losses)), y=losses, color="#a8c8e8", linewidth=1, label="loss brute")
    sns.lineplot(x=range(len(losses)), y=mean, color="#1a3a6b", linewidth=2, label=f"moyenne mobile ({window})")
    plt.xlabel("Step")
    plt.ylabel("Cross-entropy")
    plt.title("Train loss")
    plt.show()


def evaluate(model, sequences, n_batches, batch_size, device, rng=None):
    """Loss moyenne sur n_batches tirés du corpus de validation"""

    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for _ in range (n_batches):
            x, y = get_batch(sequences, batch_size, rng)
            _, loss = model(x.to(device), y.to(device))
            val_loss += loss.item()

    model.train()
    return val_loss/n_batches   


if __name__ == "__main__":

    cfg = DEV

    schedule = TrainConfig()

    B = 40

    n_steps = 200

    corpus = "train_fr"

    train_loss = train(cfg, schedule, B, n_steps, corpus, k=4)

    plot_loss(train_loss)


