import pytest
import torch
import dataclasses
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
    weights, indices, _ = moe.router(x)                                   # même router, les mêmes choix
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
    set_seed(42)
    transformer = Transformer(TEST)
    _, loss = transformer(pre_x, y)
    assert loss is not None
    loss.backward()
    for p in transformer.parameters(): assert p.grad is not None
    assert count_params(cfg) == 1079712
    moe_chi = MoE(TEST)
    with torch.no_grad():
        moe_chi.router.routing.weight.zero_()
        _ = moe_chi(x)
    assert abs(moe_chi.aux_loss - 1.0) < 1e-6
    _ = moe(x)
    moe.aux_loss.backward()
    for p in moe.router.parameters(): assert p.grad is not None
    set_seed(42)
    moe_bis = MoE(TEST)
    with torch.no_grad():
        moe_bis.router.routing.weight.zero_()
        moe_bis.router.routing.weight[0,0] = 10.0
        x_bis = torch.zeros(B, T, TEST.d_model)
        x_bis[...,0] = 1.0
        _ = moe_bis(x_bis)
    assert moe_bis.aux_loss > 1.5
    set_seed(42)
    transformer_bis = Transformer(dataclasses.replace(TEST, aux_loss_coef=0.01))
    _, loss_alpha = transformer_bis(pre_x, y)
    sum_aux = sum(layer.mlp.aux_loss for layer in transformer_bis.layers)
    assert torch.allclose(loss_alpha - loss, sum_aux * 0.01, atol=1e-6)
    moe_alpha= MoE(dataclasses.replace(TEST, aux_loss_coef=0.01))
    _ = moe_alpha(x)
    aux, f_max = moe_alpha.routing_stats()
    assert isinstance(aux, float) and isinstance(f_max, float) 
    assert abs(aux - moe_alpha.aux_loss.item()) < 1e-6
    assert 1 / TEST.n_experts <= f_max <= 1.0 + 1e-9
    _, f_max2 = moe_bis.routing_stats()
    assert f_max2 == pytest.approx(0.5)
    assert not hasattr(TransformerBlock(TEST_E1_K1).mlp, "routing_stats")



    