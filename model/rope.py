#Implémentation de RoPE pour l'encodage des positions : choix des paires par moitiés (usuellement theta=10000)
import torch 
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Encodage de la position."""

    def __init__(self, head_dim: int, max_len: int, theta: int):
        super().__init__()
        inv_freq = torch.tensor([theta**(-i / head_dim) for i in torch.arange(0, head_dim, 2).float()])
        positions = torch.arange(max_len)
        angles = torch.outer(positions.float(), inv_freq)
        cat_angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", torch.cos(cat_angles), persistent=False) # Buffer signifie que ce n'est pas un paramètre à apprender + persistent = False permet de ne pas le sauvegarder aux checkpoints car il est recalculable.
        self.register_buffer("sin", torch.sin(cat_angles), persistent=False)

    def forward(self, start: int, window_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[start:start + window_length], self.sin[start: start + window_length]


def apply_rope(x, cos: torch.Tensor, sin: torch.Tensor):
    return x * cos + rotate_half(x) * sin