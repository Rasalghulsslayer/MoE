import pytest
import torch
from torch import nn
from model.config import ModelConfig
from model.layers import RMSNorm, FeedForward
from utils.runtime import set_seed

@pytest.fixture
def x():
    set_seed(0)
    return torch.randn(2, 5, 16)

@pytest.fixture
def rmsn(): 
    return RMSNorm(d_model=16)

@pytest.fixture
def rmsn_ref():
    return torch.nn.RMSNorm(16, 1e-5)

def test_rmsn_shape_remains(x, rmsn):
    assert rmsn(x).shape == x.shape

def test_rmsn_works(x, rmsn):
    y = rmsn(x)
    rms = torch.sqrt(torch.mean(y**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

def test_rmsn_invariance(x, rmsn):
    assert torch.allclose(rmsn(3 * x), rmsn(x), atol=1e-4)

def test_rmsn_ref(x, rmsn, rmsn_ref):
    assert torch.allclose(rmsn(x), rmsn_ref(x), atol=1e-6)

def test_rmsn_num_param(rmsn):
    assert sum(p.numel() for p in rmsn.parameters()) == rmsn.d_model

def test_rmsn_keeps_dtype(x, rmsn):
    assert rmsn(x).dtype == x.dtype

def test_embedding():
    set_seed(0)
    emb = nn.Embedding(50, 16)
    ids = torch.tensor([[5,42,5]])
    out = emb(ids)

    assert out.shape == (1, 3, 16)
    assert torch.equal(out[0,0], out[0,2])
    assert torch.equal(out[0,0], emb.weight[5])
    assert not torch.equal(out[0,0], out[0,1])

B = 2
T = 10

@pytest.fixture
def cfg():
    return ModelConfig(d_model=32, n_heads=4, n_kv_heads=2, sliding_window=4)

@pytest.fixture
def y(cfg):
    set_seed(0)
    return torch.randn(B, T, cfg.d_model)

@pytest.fixture
def ffn(cfg):
    set_seed(0)
    return FeedForward(cfg)

def test_ffn_shape_remains(y, ffn):
    assert ffn(y).shape == y.shape

def test_ffn_num_param(ffn, cfg):
    assert sum(p.numel() for p in ffn.parameters()) == 3 * cfg.d_model * cfg.ffn_dim

def test_ffn_all_params_receive_grad(ffn, y):
    ffn(y).sum().backward()
    for name, p in ffn.named_parameters():
        assert p.grad is not None, f"{name} ne reçoit aucun gradient"

def test_ffn_gate_role(ffn, y):
    nn.init.zeros_(ffn.gate_proj.weight)
    assert torch.equal(ffn(y), torch.zeros(y.shape))

def test_ffn_position_independance(y, ffn, cfg):
    set_seed(0)
    y_dif = torch.cat((y[:, :5,:], torch.randn(y.shape[0], 1, cfg.d_model), y[:, 6:,:]), dim=1)
    assert torch.allclose(ffn(y_dif)[:, :5:, :], ffn(y)[:, :5:, :], atol=1e-6)
    assert not torch.allclose(ffn(y_dif)[:, 5, :], ffn(y)[:, 5, :], atol=1e-6)
     
