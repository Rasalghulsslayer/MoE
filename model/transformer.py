import torch
from torch import nn
from torch.nn import functional as F
from model.attention import Attention, KVCache
from model.layers import RMSNorm, FeedForward
from model.config import ModelConfig
from model.rope import RotaryEmbedding


class TransformerBlock(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.d_model)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.d_model)
        self.mlp = FeedForward(cfg)

    def forward(self, x, cos, sin, cache=None):
        x_post_att = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)
        x_post_mlp = x_post_att + self.mlp(self.post_attention_layernorm(x_post_att))
        return (x_post_mlp)

    
class Transformer(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, )
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights) # Initialisation des couches linéaires et des embeddings
        residual_std = 0.02 / ((2 * cfg.n_layers)**0.5)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"): # Initialisation des couches écrivant dans le flux résiduel
                nn.init.normal_(p, mean=0.0, std=residual_std)

        
    def _init_weights(self, module: nn.Module) -> None:
        """Loi normale (0, 0.02) pour toutes les couches linéaires et les embeddings"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None, caches=None):

        B, T = idx.shape

        start = caches[0].length if caches else 0
        assert start + T <= self.cfg.context_length, "Contexte trop long."
        
        cos, sin = self.rope(start, T)
        x = self.embed_tokens(idx)
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, cos, sin, cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1)) if targets is not None else None
        return(logits, loss)


    def new_caches(self, batch_size, device=None, dtype=torch.float32):
        caches = []
        for _ in range(self.cfg.n_layers):
            caches.append(KVCache(batch_size, self.cfg, device, dtype))
        return caches

