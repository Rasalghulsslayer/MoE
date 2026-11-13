from model.config import DEV
from training.loop import train

def test_loop_train():
    loss = train(DEV, 1, 10, "train_fr", device_pref="cpu")
    assert loss[-1] < loss[0]