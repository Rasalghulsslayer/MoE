import os
from pathlib import Path
import torch
from datetime import datetime, timezone
import shutil

CHECKPOINT_PATH = Path("artifacts/checkpoint")

def save_checkpoint(i: int, model, optimizer, scheduler, rng, train_loss, cfg, schedule, save_every, path: Path = CHECKPOINT_PATH):
    """Sauvegarde l'état actuel pour reprendre l'entrainement"""
    latest = path / "latest.pt"
    archive = path / f"{datetime.now(timezone.utc).isoformat()}-checkpoint.pt"
    path.mkdir(parents=True, exist_ok=True)

    data = {
            "step": i,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_numpy": rng.bit_generator.state,
            "rng_torch": torch.get_rng_state(),
            "train_loss": train_loss,
            "config": cfg,
            "train_config": schedule,
            "datetime": datetime.now(timezone.utc).isoformat()
        }
    tmp = latest.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    os.replace(tmp, latest)

    if i % save_every == 0:
        archive = path / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}-checkpoint.pt"
        shutil.copyfile(latest, archive) # quasi gratuit comparé à torch.save

    

def load_checkpoint(file: str, device=None):
    """Recharge un checkpoint."""
    return torch.load(CHECKPOINT_PATH / file, map_location="cpu", weights_only=False)