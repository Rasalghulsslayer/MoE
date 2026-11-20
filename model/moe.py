import torch
from torch import nn
from model.config import ModelConfig

class Router(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_exp = cfg.n_experts
        self.top_k = cfg.top_k
        self.rout = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x):
        logits = self.rout(x)
        pre_weights, index = torch.topk(logits, self.top_k, dim=-1)
        return (torch.softmax(pre_weights, dim=-1), index)
