import pytest 
import torch
import math
from dataclasses import replace
from utils.runtime import set_seed
from model.transformer import Transformer
from model.config import ModelConfig, count_params, DEV, RUN

B = 2

@pytest.fixture
def cfg():
    return ModelConfig(vocab_size=1000, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=64, context_length=64, sliding_window=16)

@pytest.fixture
def model(cfg):
    set_seed(0)
    return Transformer(cfg)

@pytest.fixture
def x(cfg):
    set_seed(0)
    return torch.randint(0, cfg.vocab_size, (B, cfg.context_length))

@pytest.fixture
def y(cfg):
    set_seed(1)
    return torch.randint(0, cfg.vocab_size, (B, cfg.context_length))


def test_transformer_params(cfg, model):
    model_untied = Transformer(replace(cfg, tie_embeddings=False))
    model_dev = Transformer(DEV)
    model_run = Transformer(RUN)
    num_params_model = sum(p.numel() for p in model.parameters())
    num_params_model_untied = sum(p.numel() for p in model_untied.parameters())
    num_params_DEV = sum(p.numel() for p in model_dev.parameters())
    num_params_RUN = sum(p.numel() for p in model_run.parameters())
    assert num_params_model == count_params(cfg)
    assert num_params_model_untied == count_params(replace(cfg, tie_embeddings=False))
    assert num_params_DEV == count_params(DEV)
    assert num_params_RUN == count_params(RUN)

def test_transformer_weight_tying_works(model):
    assert model.lm_head.weight is model.embed_tokens.weight

def test_transformer_initial_loss(cfg, x, y, model):
    (_, loss) = model(x, targets=y)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.2

def test_init_std():
    cfg = ModelConfig(vocab_size=1000, d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, ffn_dim=256)
    model = Transformer(cfg)
    layer = model.layers[0]
    assert abs(layer.mlp.gate_proj.weight.std().item() - 0.02) < 0.002
    expected = 0.02 / ((2 * cfg.n_layers)**0.5)
    assert abs(layer.self_attn.o_proj.weight.std().item() - expected) < 0.002
    assert torch.equal(layer.input_layernorm.weight, torch.ones(cfg.d_model))

def test_transformer_causality(cfg, x, model):
    x_diff = torch.cat((x[:, :5], torch.randint(0, cfg.vocab_size, (B, 1)), x[:, 6:]), dim=-1)
    assert torch.allclose(model(x)[0][:,:5,:], model(x_diff)[0][:,:5,:],  atol=1e-6)
    assert not torch.allclose(model(x)[0][:,5:,:], model(x_diff)[0][:,5:,:],  atol=1e-6)

def test_transformer_all_params_receive_grad(model, x, y):
    model(x, y)[1].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} ne reçoit aucun gradient"

def test_model_can_overfit_one_batch(model, x, y):
    set_seed(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(200):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5

