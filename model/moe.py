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
        probs = torch.softmax(logits, dim=-1)
        pre_weights, index = torch.topk(logits, self.top_k, dim=-1)
        return (torch.softmax(pre_weights, dim=-1), index, probs)


class MoE(nn.Module):
    """Bloc complet de Mixture of Experts"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg)
        self.experts = nn.ModuleList(
            [FeedForward(cfg) for _ in range(cfg.n_experts)]
        )
        self._f = torch.zeros(cfg.n_experts)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        cfg = self.cfg
        B, T, d_model = x.shape
        x_flat = x.view(B * T, d_model)
        weights, index, probs = self.router(x_flat)
        weights_flat = weights.view(B * T * cfg.top_k)
        index_flat = index.view(B * T * cfg.top_k)
        f = torch.bincount(index_flat, minlength=cfg.n_experts) / (B * T * cfg.top_k)
        self._f = f.detach()
        p = torch.mean(probs, dim=0)
        self.aux_loss = cfg.n_experts * torch.sum(f * p)
        lines = torch.arange(B*T).repeat_interleave(cfg.top_k)
        out = torch.zeros_like(x_flat)
        for e in range(cfg.n_experts):
            mask = (index_flat == e)
            if mask.any():
                lines_e = lines[mask]
                weights_e = weights_flat[mask]
                out[lines_e] += weights_e.unsqueeze(-1) * self.experts[e](x_flat[lines_e])
        return out.view(B, T, d_model)

    def routing_stats(self):
        f = self._f
        f_max = f.max()
        return self.aux_loss.item(), f_max.item()



