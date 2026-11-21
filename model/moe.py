import torch
from torch import nn
from model.config import ModelConfig
from model.layers import FeedForward

class Router(nn.Module):
    """Module permettant de sélectionner les experts"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_exp = cfg.n_experts
        self.top_k = cfg.top_k
        self.routing = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x):
        logits = self.routing(x)
        pre_weights, index = torch.topk(logits, self.top_k, dim=-1)
        return (torch.softmax(pre_weights, dim=-1), index)


class MoE(nn.Module):
    """Bloc complet de Mixture of Experts"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg)
        self.experts = nn.ModuleList(
            [FeedForward(cfg) for _ in range(cfg.n_experts)]
        )

    def forward(self, x):
        cfg = self.cfg
        B, T, d_model = x.shape
        x = x.view(B * T, d_model)
        weights, index = self.router(x)
        out = torch.zeros_like(x)
        for i in range (B * T):
            for j in range(cfg.top_k):
                out[i] += weights[i,j] * self.experts[index[i,j]](x[i,:])
        return out.view(B, T, d_model)

