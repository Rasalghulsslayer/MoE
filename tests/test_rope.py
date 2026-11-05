import pytest
import torch
from utils.runtime import set_seed
from model.rope import RotaryEmbedding, apply_rope, rotate_half

HEAD_DIM = 64      
MAX_LEN = 1024
THETA = 10_000.0

@pytest.fixture
def x():
    set_seed(0)
    return torch.randn(2, 3, 10, HEAD_DIM)

@pytest.fixture
def rope(): 
    return RotaryEmbedding(HEAD_DIM, MAX_LEN, THETA)

def test_rope_shape_remains(x, rope):
    w_l = x.shape[-2] #seq_len
    cos, sin = rope(0, w_l)
    assert apply_rope(x, cos, sin).shape == x.shape

def test_rope_norm_remains(x, rope):
    w_l = x.shape[-2] 
    cos, sin = rope(0, w_l)
    assert torch.allclose(torch.norm(apply_rope(x, cos, sin)), torch.norm(x), atol=1e-6)

def test_rope_pos_0(x, rope):
    w_l = x.shape[-2]
    cos, sin = rope(0, w_l)
    assert torch.allclose(apply_rope(x, cos, sin)[...,0,:], x[...,0,:], atol=1e-6)

def test_rope_relativity(rope):
    set_seed(0)
    q, k = torch.randn(HEAD_DIM), torch.randn(HEAD_DIM)
    m, n = 990, 1000
    assert torch.allclose(torch.dot(apply_rope(q, rope(m, 1)[0][0], rope(m,1)[1][0]), apply_rope(k, rope(n, 1)[0][0], rope(n,1)[1][0])), torch.dot(apply_rope(q, rope(m+5, 1)[0][0], rope(m+5,1)[1][0]), apply_rope(k, rope(n+5, 1)[0][0], rope(n+5,1)[1][0])), atol=1e-6)

def test_rotate_half():
    assert torch.equal(rotate_half(torch.tensor([1., 2., 3., 4.])), torch.tensor([-3., -4., 1., 2.]))

