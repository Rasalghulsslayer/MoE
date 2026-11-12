# Le module d'attention
import torch
from torch import nn
from torch.nn import functional as F
from model.config import ModelConfig
from model.rope import apply_rope


class KVCache:
    """Tampon circulaire de k et v de taille fixe `window`."""

    def __init__(self, batch_size: int, cfg: ModelConfig, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
        self.k = torch.zeros(batch_size, cfg.n_kv_heads, cfg.sliding_window, cfg.head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.window = cfg.sliding_window
        self.length = 0  # nombre de tokens déjà rangés / position du prochain token

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        T = k_new.shape[2]
        positions = torch.arange(self.length, self.length + T, device=self.k.device)
        if T > self.window:  # seules les `window` dernières positions doivent rester
            positions = positions[-self.window:]
            k_new, v_new = k_new[:, :, -self.window:], v_new[:, :, -self.window:]
        slots = positions % self.window
        self.k[:, :, slots] = k_new
        self.v[:, :, slots] = v_new
        self.length += T

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = min(self.length, self.window)
        return self.k[:, :, :n], self.v[:, :, :n]


def sliding_window_mask(T: int, window: int, device: torch.device | None = None) -> torch.Tensor:
    return (torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)) & ~ torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-window))


class Attention(nn.Module):
    """Attention layer"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.q_proj = nn.Linear(cfg.d_model, cfg.head_dim * cfg.n_heads, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.head_dim * cfg.n_kv_heads, bias=False )
        self.v_proj = nn.Linear(cfg.d_model, cfg.head_dim * cfg.n_kv_heads, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.cfg = cfg
        self.window = cfg.sliding_window


    def forward(self, x, cos, sin, cache: KVCache | None = None, use_sdpa=False):
        cfg = self.cfg

        B = x.shape[0]
        T = x.shape[1]

        q = self.q_proj(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1,2)
        k = (x @ self.k_proj.weight.T).view(B, T, cfg.n_kv_heads, -1).transpose(1,2)
        v = (torch.matmul(x, self.v_proj.weight.T)).view(B, T, cfg.n_kv_heads, -1).transpose(1,2)

        roped_q = apply_rope(q, cos, sin)
        roped_k = apply_rope(k, cos, sin)

        if cache is None: # Sans KVCache

            filled_roped_k = roped_k.repeat_interleave(cfg.n_heads // cfg.n_kv_heads, dim=1)
            filled_v = v.repeat_interleave(cfg.n_heads // cfg.n_kv_heads, dim=1)

            mask = sliding_window_mask(T, self.window, x.device)

            if use_sdpa == True:
                values = F.scaled_dot_product_attention(roped_q, filled_roped_k, filled_v, attn_mask=mask)
            else:
                scores = (roped_q @ filled_roped_k.transpose(-2, -1)) * ((cfg.head_dim)**(-0.5))
                masked_scores = torch.where(mask, scores, float('-inf'))
                values = torch.softmax(masked_scores.float(), dim=-1).type_as(x) @ filled_v

        elif cache.length == 0: # Initialisation du KVCache

            cache.update(roped_k, v) # Le cache n'est pas utilisé pour les calculs.

            filled_roped_k = roped_k.repeat_interleave(cfg.n_heads // cfg.n_kv_heads, dim=1)
            filled_v = v.repeat_interleave(cfg.n_heads // cfg.n_kv_heads, dim=1)

            mask = sliding_window_mask(T, self.window, x.device)
            
            if use_sdpa == True:
                values = F.scaled_dot_product_attention(roped_q, filled_roped_k, filled_v, attn_mask=mask)
            else:
                scores = (roped_q @ filled_roped_k.transpose(-2, -1)) * ((cfg.head_dim)**(-0.5))
                masked_scores = torch.where(mask, scores, float('-inf'))
                values = torch.softmax(masked_scores.float(), dim=-1).type_as(x) @ filled_v


        else: # En plein dans le KVCache
            assert (T==1), "Mauvaise entrée pour KVCache, en attente d'une seul token"

            cache.update(roped_k,v) # Le cache est utilisé pour les calculs, donc remonté au début car le token doit pouvoir se voir.

            k_all, v_all = cache.get()

            filled_k_all = k_all.repeat_interleave(cfg.n_heads // cfg.n_kv_heads, dim=1)
            filled_v_all = v_all.repeat_interleave(cfg.n_heads // cfg.n_kv_heads, dim=1)

            if use_sdpa == True:
                values = F.scaled_dot_product_attention(roped_q, filled_k_all, filled_v_all, attn_mask=None)
            else:
                scores = (roped_q @ filled_k_all.transpose(-2, -1)) * ((cfg.head_dim)**(-0.5))
                values = torch.softmax(scores.float(), dim=-1).type_as(x) @ filled_v_all

        out = self.o_proj(values.transpose(1,2).contiguous().view(B, T, cfg.d_model))

        return out
    