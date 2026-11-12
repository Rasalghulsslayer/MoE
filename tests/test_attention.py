import torch
import pytest
from torch import nn
from model.config import ModelConfig
from utils.runtime import set_seed
from model.rope import RotaryEmbedding
from model.attention import Attention, sliding_window_mask, KVCache

B = 2
T = 10

@pytest.fixture
def cfg():
    return ModelConfig(d_model=32, n_heads=4, n_kv_heads=2, sliding_window=4)

@pytest.fixture
def x(cfg):
    set_seed(0)
    return torch.randn(B, T, cfg.d_model)

@pytest.fixture
def rope():
    return RotaryEmbedding(head_dim=8, max_len=10, theta=10_000)

@pytest.fixture
def attention(cfg):
    return Attention(cfg)

def test_attention_shape(x, attention, rope):
    cos, sin = rope(0, 10)
    assert attention(x, cos, sin).shape == x.shape

def test_attention_causality(x, attention, rope, cfg):
    cos, sin = rope(0, 10)
    set_seed(0)
    x_dif = torch.cat((x[:, :5,:], torch.randn(B, 5, cfg.d_model)), dim=1)
    assert torch.allclose(attention(x_dif, cos, sin)[:, :5, :], attention(x, cos, sin)[:, :5, :], atol=1e-6)
    assert not torch.allclose(attention(x_dif, cos, sin)[:, 5:, :], attention(x, cos, sin)[:, 5:, :], atol=1e-6)

def test_attention_pytorch_equality(x, attention, rope):
    cos, sin = rope(0,10)
    assert torch.allclose(attention(x, cos, sin), attention(x, cos, sin, use_sdpa=True), atol=1e-6)

def test_attention_one_token(attention, rope, cfg):
    set_seed(0)
    y = torch.randn(B, 1, cfg.d_model)
    cos, sin = rope(0,1)
    out = attention(y, cos, sin)

    v = attention.v_proj(y).view(2, 1, 2, 8)            
    v = v.repeat_interleave(2, dim=2)              
    expected = attention.o_proj(v.reshape(2, 1, 32))    

    assert out.shape == (2, 1, 32)
    assert torch.allclose(out, expected, atol=1e-6)

def test_attention_count_params(attention, cfg):
    assert sum(p.numel() for p in attention.parameters()) == cfg.attention_dim

def test_attention_all_params_receive_grads(attention, rope, x):
    cos, sin = rope(0, x.shape[1])
    attention(x, cos, sin).sum().backward()
    for name, p in attention.named_parameters():
        assert p.grad is not None, f"{name} ne reçoit aucun gradient"

def test_sliding_window_mask():
    expected = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(sliding_window_mask(5, 2), expected)

def test_large_window_is_causal():
    T = 6
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.equal(sliding_window_mask(T, window=10), causal)

def test_sliding_window_mask_locality(x, attention, rope, cfg):
    cos, sin = rope(0, 10)
    set_seed(0)
    x_dif = torch.cat((x[:, :2,:], torch.randn(B, 1, cfg.d_model), x[:, 3:,:]), dim=1)
    assert torch.allclose(attention(x_dif, cos, sin)[:, 6:, :], attention(x, cos, sin)[:, 6:, :], atol=1e-6)
    assert not torch.allclose(attention(x_dif, cos, sin)[:, 5, :], attention(x, cos, sin)[:, 5, :], atol=1e-6)
    assert not torch.allclose(attention(x_dif, cos, sin)[:, 4, :], attention(x, cos, sin)[:, 4, :], atol=1e-6)
    assert not torch.allclose(attention(x_dif, cos, sin)[:, 3, :], attention(x, cos, sin)[:, 3, :], atol=1e-6)
    assert not torch.allclose(attention(x_dif, cos, sin)[:, 2, :], attention(x, cos, sin)[:, 2, :], atol=1e-6)
    assert torch.allclose(attention(x_dif, cos, sin)[:, 1, :], attention(x, cos, sin)[:, 1, :], atol=1e-6)
    assert torch.allclose(attention(x_dif, cos, sin)[:, 0, :], attention(x, cos, sin)[:, 0, :], atol=1e-6)

def test_sliding_window_mask_pytorch_equality(x, attention, rope):
    cos, sin = rope(0,10)
    assert torch.allclose(attention(x, cos, sin), attention(x, cos, sin, use_sdpa=True), atol=1e-6)

@pytest.mark.parametrize("prefill", [1, 3, 6])
def test_kv_cache_matches_full_sequence(attention, cfg, rope, x, prefill):
    cos, sin = rope(0, T)
    full = attention(x, cos, sin)

    cache = KVCache(B, cfg)
    cos, sin = rope(0, prefill)
    outs = [attention(x[:, :prefill], cos, sin, cache=cache)]
    for p in range(prefill, T):
        cos, sin = rope(p, 1)
        outs.append(attention(x[:, p:p + 1], cos, sin, cache=cache))
    incremental = torch.cat(outs, dim=1)

    assert torch.allclose(incremental, full, atol=1e-5)