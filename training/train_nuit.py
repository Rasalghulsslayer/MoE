#Run nuit DEV_MOE : recette du jumeau carotte, une seule variable changée : 3000 steps.
from model.config import DEV_MOE
from training.train_config import TrainConfig
from training.loop import train


def main() -> None:
    """Lance le run nuit : 3000 steps, jumeau carotte (alpha=0,01), CPU."""
    schedule = TrainConfig(base_lr=3e-4, warmup_steps=20, max_steps=3000)   # l'horizon suit le run, pas l'inverse
    train_loss, val_loss, t = train(DEV_MOE, schedule, batch_size=16, n_steps=3000,
                                    corpus="fr", k=1, device_pref="cpu")


if __name__ == "__main__":
    main()