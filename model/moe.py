import torch
from torch import nn
from model.config import ModelConfig
from model.generate import filter_top_k

class Router(nn.Module):

    def __init__(self, cfg: ModelConfig, n_exp: int, top_k: int):
        super().__init__()
        self.cfg = cfg
        self.n_exp = n_exp
        self.top_k = top_k
        self.rout = nn.Linear(cfg.d_model, n_exp, bias=False)

    def forward(self, x):
        logits = self.rout(x)
        pre_weights, index = torch.topk(logits, self.top_k, dim=-1)
        return (torch.softmax(pre_weights, dim=-1, dtype=float), index)
