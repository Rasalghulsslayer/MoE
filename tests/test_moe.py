import pytest
import torch
from utils.runtime import set_seed
from model.moe import Router
from model.config import TEST

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

def test_router_props(x, cfg, router):
    print
    assert router(x)[0].shape == (B, T, cfg.top_k)
    assert router(x)[1].shape == (B, T, cfg.top_k)
    assert torch.all(router(x)[0] > 0) and  torch.all(router(x)[0] <= 1) 
    assert torch.allclose(torch.sum(router(x)[0], -1), torch.ones(B, T), atol=1e-6)
    for i in range (router.top_k): assert torch.all(0 < router(x)[1][...,i]) and torch.all(router(x)[1][...,i] <= cfg.n_experts - 1)
    for i in range (router.top_k):
        for j in range (i+1, router.top_k): assert torch.all(router(x)[1][...,i] != router(x)[1][...,j])
    router(x)[0].sum().backward()
    for _, p in router.named_parameters():
            assert p.grad is not None
    router_bis = Router(cfg, 1, 1)
    assert torch.all(router_bis(x)[0] == 1.0)
    assert torch.all(router_bis(x)[1] == 0)