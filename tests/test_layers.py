import pytest
import torch
from torch import nn
from model.layers import RMSNorm
from utils.runtime import set_seed


@pytest.fixture
def x():
    set_seed(0)
    return torch.randn(2, 5, 16)

RMSN = RMSNorm(d=16)

RMSN_ref = torch.nn.RMSNorm(16, 1e-5)

def test_rmsn_shape_remains(x):
    assert RMSN(x).shape == x.shape

def test_rmsn_works(x):
    y = RMSN(x)
    rms = torch.sqrt(torch.mean(y**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

def test_rmsn_invariance(x):
    assert torch.allclose(RMSN(3 * x), RMSN(x), atol=1e-4)

def test_rmsn_ref(x):
    assert torch.allclose(RMSN(x), RMSN_ref(x), atol=1e-6)

def test_rmsn_num_param():
    assert sum(p.numel() for p in RMSN.parameters()) == RMSN.d

def test_rmsn_keeps_dtype(x):
    assert RMSN(x).dtype == x.dtype

def test_embedding():
    set_seed(0)
    emb = nn.Embedding(50, 16)
    ids = torch.tensor([[5,42,5]])
    out = emb(ids)

    assert out.shape == (1, 3, 16)
    assert torch.equal(out[0,0], out[0,2])
    assert torch.equal(out[0,0], emb.weight[5])
    assert not torch.equal(out[0,0], out[0,1])



