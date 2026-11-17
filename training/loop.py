from model.config import ModelConfig, DEV
from model.transformer import Transformer
from training.dataset import get_batch, make_sequences, load_tokens
from utils.runtime import set_seed, get_device
from training.schedule import get_lr
from training.train_config import TrainConfig
from training.checkpoint import load_checkpoint, save_checkpoint, CHECKPOINT_PATH
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
        data = load_checkpoint(resume_from, device=device)

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
        

    tokens = load_tokens(f"data/tokens/{corpus}.bin")
    sequences = make_sequences(tokens, cfg.context_length)

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
            print(f"Perte actuelle : {train_loss[-1]}")
            print(f"Temps d'entrainement du dernier micro-batch : {t_2 - t_step} secondes")
            t_step = time.perf_counter()
            print(f"Temps d'entrainement total : {t_2 - t_1} secondes")
            print("\n")
            save_checkpoint(i, model, optimizer, scheduler, rng, train_loss, cfg, schedule, 1)


    return train_loss



def plot_loss(losses, window=20):
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

if __name__ == "__main__":

    cfg = DEV

    schedule = TrainConfig()

    B = 40

    n_steps = 200

    corpus = "train_fr"

    train_loss = train(cfg, schedule, B, n_steps, corpus, k=4)

    plot_loss(train_loss)


