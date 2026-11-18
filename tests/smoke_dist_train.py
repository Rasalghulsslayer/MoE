"""Mini-entraînement distribué : 2 processus, 4 steps, config TEST, k=1."""
from model.config import TEST
from training.train_config import TrainConfig
from training.loop import train

if __name__ == "__main__":
    train(TEST, TrainConfig(), batch_size=4, n_steps=4, corpus="fr", k=1)