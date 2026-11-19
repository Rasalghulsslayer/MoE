from training.train_config import TrainConfig
from model.config import DEV
from training.loop import train, plot_loss

if __name__ == "__main__":

    # dist = setup_distributed()
    # print(dist)
    # if dist is not None:
    #     torch.distributed.destroy_process_group()

    cfg = DEV
    schedule = TrainConfig()
    B = 16
    k = 1
    n_steps = 12
    device_pref = "cpu"
    corpus = "fr"
    
    train_loss, val_loss, mean_t_step = train(cfg, schedule, B, n_steps, corpus, k=k, device_pref=device_pref)
    print(f"Temps moyen par step : {mean_t_step}")
    plot_loss(train_loss, val_losses=val_loss)