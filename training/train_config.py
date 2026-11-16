from dataclasses import dataclass

@dataclass
class TrainConfig:
    base_lr: float = 3e-4
    min_lr: float = base_lr / 10
    warmup_steps: int = 20
    max_steps: int = 200

