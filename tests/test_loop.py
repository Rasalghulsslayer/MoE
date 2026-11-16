import pytest
from model.config import TEST, DEV
from training.loop import train, train_step, make_sequences
from training.dataset import get_batch
from model.transformer import Transformer
from training.train_config import TrainConfig
import numpy as np
from utils.runtime import set_seed
import torch

@pytest.fixture
def schedule():
    return TrainConfig()

def test_loop_train_canari():
    set_seed(42)
    B=1

    model = Transformer(TEST)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    rng = np.random.default_rng(42)
    tokens = rng.choice(TEST.vocab_size, 100, replace=False)
    sequence = make_sequences(tokens, 50)
    x, y = get_batch(sequence, B, rng)

    loss=[]
    for _ in range (50):
        optimizer.zero_grad()
        loss.append(train_step(model, x, y))
        optimizer.step()
    print(loss)
    assert loss[-1] < loss[0]
    

def test_loop_train(schedule):
    loss = train(DEV, schedule, 1, 10, "train_fr", device_pref="cpu")
    assert loss[-1] < loss[0]

def test_loop_mini_batch_processing(schedule):
    set_seed(42)
    rng = np.random.default_rng(42)
    
    B, T = (2, 50)

    tokens = rng.choice(TEST.vocab_size, 200, replace=False)
    sequences = make_sequences(tokens, T)
    x, y = get_batch(sequences, B, rng)

    model = Transformer(TEST)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.base_lr)

    optimizer.zero_grad()
    train_step(model, x, y)
    grads_A = {name: p.grad.clone() for name, p in model.named_parameters()}

    x1, x2 = torch.split(x, int(B/2))
    y1, y2 = torch.split(y, int(B/2))
    optimizer.zero_grad()
    train_step(model, x1, y1, k=2)
    train_step(model, x2, y2, k=2)
    grads_B = {name: p.grad.clone() for name, p in model.named_parameters()}

    assert all(torch.allclose(grads_A[name],grads_B[name], atol=1e-5) for name, _ in model.named_parameters())