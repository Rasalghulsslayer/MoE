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
    return Router(cfg, n_exp=4, top_k=2)

@pytest.fixture
def x(cfg):
    set_seed(42)
    return torch.randn(B, T, cfg.d_model)

def test_router_props(x, cfg, router):
    print
    assert router(x)[0].shape == (B, T, router.top_k)
    assert router(x)[1].shape == (B, T, router.top_k)
    assert torch.any(router(x)[0] > 0) and  torch.any(router(x)[0] <= 1) 
    assert torch.allclose(torch.sum(router(x)[0], dim=-1), torch.ones(B, T, dtype=float), atol=1e-6)
    assert torch.any(0 < router(x)[1]) and torch.any(router(x)[1] <= router.n_exp - 1)
    for i in range (router.top_k):
        for j in range (i+1, router.top_k): assert torch.any(router(x)[1][i] != router(x)[1][j])
    assert [p.grad is not None for p in router.parameters()] 
    router_bis = Router(cfg, 1, 1)
    assert torch.any(router_bis(x)[0] == 1.0)
    assert torch.any(router_bis(x)[1] == 0)