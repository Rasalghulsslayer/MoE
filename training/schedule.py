import math
from training.train_config import TrainConfig

def get_lr(step: int, schedule: TrainConfig) -> float:
    base_lr = schedule.base_lr
    min_lr = schedule.min_lr
    warmup_steps = schedule.warmup_steps  
    max_steps = schedule.max_steps

    if warmup_steps == 0:
        return base_lr
    else : 
        if step < warmup_steps:
            return min_lr + (base_lr - min_lr) * ((step + 1) / warmup_steps )
        elif warmup_steps <= step <= max_steps:
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step))
        elif step < max_steps:
            return min_lr


