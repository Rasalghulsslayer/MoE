import math
from training.train_config import TrainConfig

def get_lr(step: int, schedule: TrainConfig) -> float:
    """Donne le learning rate en foncion du planning prévu."""
    base_lr = schedule.base_lr
    min_lr = schedule.min_lr
    warmup_steps = schedule.warmup_steps  
    max_steps = schedule.max_steps

    if step > max_steps:
                return min_lr
    elif warmup_steps <= step <= max_steps:
        t = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    else:
        return min_lr + (base_lr - min_lr) * ((step) / warmup_steps)
        
        


