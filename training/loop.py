from model.config import ModelConfig, DEV
from model.transformer import Transformer
from training.dataset import get_batch, make_sequences, load_tokens
from utils.runtime import set_seed, get_device
from training.schedule import get_lr
from training.train_config import TrainConfig
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time

def train_step(model: Transformer, x: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    """Un unique pas d'entraînement"""
    _, loss = model(x, targets=y)
    loss = loss / k
    loss.backward()
    return loss.item()

def train(cfg: ModelConfig, schedule: TrainConfig, batch_size: int, n_steps: int, corpus: str, k: int=1, device_pref=None):
    """Boucle d'entrainement complète"""

    set_seed(42)
    rng = np.random.default_rng(42)

    device = get_device(device_pref)
    model = Transformer(cfg) 
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.base_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: get_lr(step, schedule)/schedule.base_lr)

    tokens = load_tokens(f"data/tokens/{corpus}.bin")
    sequences = make_sequences(tokens, cfg.context_length)

    t_1 = time.perf_counter()
    loss=0
    train_loss = []
    optimizer.zero_grad(set_to_none=True)

    for i in range (n_steps):

        t_step = time.perf_counter()

        x, y = get_batch(sequences, int(batch_size/k), rng)

        x = x.to(device)
        y = y.to(device)

        loss += train_step(model, x, y, k=k)

        if (i + 1) % k == 0:
            train_loss.append(loss)
            loss = 0
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            t_2 = time.perf_counter()
            print("\n")
            print(f"===Modèle en cours d'entrainement, step: {i} / {n_steps}===")
            print(f"Learning rate : {scheduler.get_last_lr()}")
            print(f"Perte actuelle : {train_loss[-1]}")
            print(f"Temps d'entrainement du dernier step : {t_2 - t_step} secondes")
            print(f"Temps d'entrainement total : {t_2 - t_1} secondes")
            print("\n")


    return train_loss



def plot_loss(losses, window=20):
    moyenne = [
        sum(losses[max(0, i - window + 1):i + 1]) / (i - max(0, i - window + 1) + 1)
        for i in range(len(losses))
    ]
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(losses)), y=losses, color="#a8c8e8", linewidth=1, label="loss brute")
    sns.lineplot(x=range(len(losses)), y=moyenne, color="#1a3a6b", linewidth=2, label=f"moyenne mobile ({window})")
    plt.xlabel("Step")
    plt.ylabel("Cross-entropy")
    plt.title("Train loss")
    plt.show()

if __name__ == "__main__":

    cfg = DEV

    schedule = TrainConfig()

    B = 10

    n_steps = 200

    corpus = "train_fr"

    train_loss = train(cfg, schedule, B, n_steps, corpus)

    plot_loss(train_loss)


