from training.train_config import TrainConfig
from model.config import DEV
import os
from training.loop import train, plot_loss


if __name__ == "__main__":

    cfg = DEV
    schedule = TrainConfig()
    B = 8
    k = 1
    n_steps = int(os.environ.get("N_STEPS", "12"))
    resume_from = "latest.pt" if os.environ.get("RESUME") == "1" else None
    device_pref = None
    corpus = "fr"
    
    train_loss, val_loss, mean_t_step = train(cfg, schedule, B, n_steps, corpus, k=k, device_pref=device_pref, resume_from=resume_from)

    plot_loss(train_loss, val_losses=val_loss)
    if os.environ.get("RANK", "0") == "0":
        print(f"Temps moyen par step : {mean_t_step}")