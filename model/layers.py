import torch
from torch import nn

class RMSNorm(nn.Module):
    """Root Mean Square Normalisation module"""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.d = d
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        ms = torch.mean(x_f.pow(2), dim=-1, keepdim=True) + self.eps
        out = self.weight * (x_f * torch.rsqrt(ms)).type_as(x)
        return out