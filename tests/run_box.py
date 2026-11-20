import os
from model.config import RUN
from training.train_config import TrainConfig
from training.loop import train

if __name__ == "__main__":
    n_steps = int(os.environ.get("N_STEPS", "100"))
    B = int(os.environ.get("B", "32"))
    resume_from = "latest.pt" if os.environ.get("RESUME") == "1" else None
    train(RUN, TrainConfig(), B, n_steps, "fr", k=4, device_pref=None, resume_from=resume_from)