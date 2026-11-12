# Le module de normalisation
import torch
from torch import nn
import torch.nn.functional as F
from model.config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Normalisation module"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        ms = torch.mean(x_f.pow(2), dim=-1, keepdim=True) + self.eps
        out = self.weight * (x_f * torch.rsqrt(ms)).type_as(x)
        return out


class FeedForward(nn.Module):
    """La couche dense du bloc Transformer"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor):
        x_gated = F.silu(self.gate_proj(x))
        x_upped = self.up_proj(x)
        x_downed = self.down_proj(x_gated * x_upped)
        return x_downed

