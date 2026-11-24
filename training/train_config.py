from dataclasses import dataclass


@dataclass
class TrainConfig:
    base_lr: float = 3e-4
    warmup_steps: int = 20
    max_steps: int = 3000

    @property
    def min_lr(self):
        return self.base_lr / 10

