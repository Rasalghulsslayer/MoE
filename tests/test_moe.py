import pytest
import torch
import torch.nn as nn
from utils.runtime import set_seed
from model.layers import FeedForward
from model.moe import Router, MoE
from model.config import TEST, TEST_E1_K1, count_params
from model.transformer import TransformerBlock, Transformer

B, T = 4, 10

@pytest.fixture
def cfg():
    return TEST

@pytest.fixture
def router(cfg):
    set_seed(42)
    return Router(cfg)

@pytest.fixture
def x(cfg):
    set_seed(42)
    return torch.randn(B, T, cfg.d_model)

@pytest.fixture
def pre_x(cfg):
    set_seed(0)
    return torch.randint(0, cfg.vocab_size, (B, T))

@pytest.fixture
def y(cfg):
    set_seed(1)
    return torch.randint(0, cfg.vocab_size, (B, T))

@pytest.fixture
def moe(cfg):
    set_seed(42)
    return MoE(cfg)


def test_router_props(x, cfg, router):
    assert router(x)[0].shape == (B, T, cfg.top_k)
    assert router(x)[1].shape == (B, T, cfg.top_k)
    assert torch.all(router(x)[0] > 0) and torch.all(router(x)[0] <= 1) 
    assert torch.allclose(torch.sum(router(x)[0], -1), torch.ones(B, T), atol=1e-6)
    for i in range (router.top_k): assert torch.all(0 <= router(x)[1][...,i]) and torch.all(router(x)[1][...,i] <= (cfg.n_experts - 1))
    for i in range (router.top_k):
        for j in range (i+1, router.top_k): assert torch.all(router(x)[1][...,i] != router(x)[1][...,j])
    router(x)[0].sum().backward()
    for p in router.parameters(): assert p.grad is not None
    router_bis = Router(TEST_E1_K1)
    assert torch.all(router_bis(x)[0] == 1.0)
    assert torch.all(router_bis(x)[1] == 0)

def moe_naive(moe, x, cfg):
    """Le MoE version 'je me fais pas chier' : E× le calcul, zéro ingénierie."""
    top_k, n_exp = cfg.top_k, cfg.n_experts
    weights, indices = moe.router(x)                                   # même router, les mêmes choix
    outs = torch.stack([expert(x) for expert in moe.experts])          # (E, B, T, d) : tout le monde calcule tout
    out = torch.zeros_like(x)
    for i in range(top_k):                                         # pour chaque position de choix i
        for e in range(n_exp):                                     # pour chaque expert e
            mask = indices[..., i] == e                                # (B, T) : les tokens dont le i-ème choix est e
            out[mask] += weights[..., i][mask, None] * outs[e][mask]   # mélange, sans ruse aucune
    return out

def test_MoE_props(x, pre_x, y, cfg, moe):
    assert moe(x).shape == x.shape
    assert torch.allclose(moe_naive(moe, x, cfg), moe(x), atol=1e-5)
    moe_ff = MoE(TEST_E1_K1)
    ff = moe_ff.experts[0]
    assert torch.allclose(moe_ff(x), ff(x), atol=1e-6)
    assert isinstance(TransformerBlock(TEST).mlp, MoE)
    assert isinstance(TransformerBlock(TEST_E1_K1).mlp, FeedForward)
    transformer = Transformer(TEST)
    _, loss = transformer(pre_x, y)
    print(transformer(pre_x, targets=y))
    assert loss is not None
    loss.backward()
    for p in transformer.parameters(): assert p.grad is not None
    assert count_params(cfg) == 1079712