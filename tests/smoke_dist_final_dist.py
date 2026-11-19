from training.train_config import TrainConfig
from model.config import DEV
from training.loop import train, plot_loss


if __name__ == "__main__":

    cfg = DEV
    schedule = TrainConfig()
    B = 8
    k = 1
    n_steps = 12
    device_pref = "cpu"
    corpus = "fr"
    
    train_loss, val_loss, mean_t_step = train(cfg, schedule, B, n_steps, corpus, k=k, device_pref=device_pref)

    print(f"Temps moyen par step : {mean_t_step}")
    plot_loss(train_loss, val_losses=val_loss)
    

    train_loss_2, val_loss_2, mean_t_step_2 = train(cfg, schedule, B, n_steps, corpus, k=k, device_pref=device_pref, resume_from="latest.pt")

    plot_loss(train_loss_2, val_losses=val_loss_2)
    print(f"Temps moyen par step : {mean_t_step_2}")