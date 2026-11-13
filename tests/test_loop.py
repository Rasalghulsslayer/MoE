from model.config import TEST, DEV
from training.loop import train, train_step, make_sequences
from training.dataset import get_batch
from model.transformer import Transformer
import numpy as np
from utils.runtime import set_seed
import torch

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
    for i in range (50):
        loss.append(train_step(model, optimizer, x, y))
    print(loss)
    assert loss[-1] < loss[0]
    




def test_loop_train():
    loss = train(DEV, 1, 10, "train_fr", device_pref="cpu")
    assert loss[-1] < loss[0]