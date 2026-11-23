import dataclasses
from training.train_config import TrainConfig
from model.config import DEV, DEV_MOE
from training.loop import train, plot_loss

if __name__ == "__main__":

    schedule = TrainConfig()
    B = 16
    n_steps = 100
    k = 1

    device_pref = "cpu"
    corpus = "fr"
    
    train_loss0, val_loss0, mean_t_step0 = train(dataclasses.replace(DEV_MOE, aux_loss_coef=0.0), schedule, B, n_steps, corpus, k=k, device_pref=device_pref)
    train_loss1, val_loss1, mean_t_step1 = train(DEV_MOE, schedule, B, n_steps, corpus, k=k, device_pref=device_pref)

    train_lossd, val_lossd, mean_t_stepd = train(DEV, schedule, B, n_steps, corpus, k=k, device_pref=device_pref)

    print("\n===BILAN===")
    print(f"val loss final alpha=0 : {val_loss0[-1]:.4f}")
    print(f"val loss final alpha=0.01 : {val_loss1[-1]:.4f}")
    print(f"chrono dense {mean_t_stepd:.3f}s/step vs MoE {mean_t_step1:.3f}s/step = {mean_t_step1/mean_t_stepd:.2f}x")
    plot_loss(train_loss0, val_losses=val_loss0)
    plot_loss(train_loss1, val_loss1)

    