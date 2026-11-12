import pytest
import torch
import math
from utils.runtime import set_seed
from model.generate import generate, filter_top_p
from model.transformer import Transformer
from model.config import ModelConfig, TEST

B = 2
T = 5

@pytest.fixture
def cfg():
    return TEST

@pytest.fixture
def x(cfg):
    set_seed(0)
    return torch.randint(0, cfg.vocab_size, (B, T))

@pytest.fixture
def model(cfg):
    set_seed(0)
    return Transformer(cfg)

def test_generate_cache(x, model):
    assert torch.equal(generate(model, x, max_new_tokens=30, temperature=0.), generate(model, x, max_new_tokens=30, temperature=0., use_cache=True))

def test_generate_shape(x, model):
    assert torch.equal(torch.tensor(generate(model, x, max_new_tokens=30, temperature=0.3).shape), torch.tensor([B,T + 30]))
    assert torch.equal(generate(model, x, max_new_tokens=30, temperature=0.3)[:, :T], x)

def test_generate_top_k(x, model):
    print(generate(model, x, max_new_tokens=30, temperature=0.5, top_k=1))
    print(generate(model, x, max_new_tokens=30, temperature=0.))
    assert torch.equal(generate(model, x, max_new_tokens=30, temperature=0.5, top_k=1), generate(model, x, max_new_tokens=30, temperature=0.))

def test_generate_top_p(x, model):
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    assert torch.allclose(filter_top_p(logits, 0.8), torch.tensor([[math.log(0.5), math.log(0.3), float('-inf'), float('-inf')]]), atol=1e-6)

