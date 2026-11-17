import pytest
import numpy as np
from utils.runtime import set_seed
from model.config import TEST
from training.loop import train, plot_loss
from training.schedule import TrainConfig

@pytest.fixture
def schedule():
    return TrainConfig()

@pytest.fixture
def cfg():
    return TEST


def test_checkpoint(cfg, schedule, corpus = "fr"):

    B = 10

    train_loss_full_training = train(cfg, schedule, B, 40, corpus, k=2)

    train(cfg, schedule, B, 20, corpus, k=2)

    train_loss_split_training = train(cfg, schedule, B, 40, corpus, k=2, resume_from="latest.pt")

    assert all(abs(train_loss_split_training[0][i] - train_loss_full_training[0][i]) < 1e-6 for i in range(len(train_loss_split_training)))






