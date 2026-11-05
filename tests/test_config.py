import pytest
from model.config import ModelConfig, RUN, DEV, count_params

def test_RUN_params():
    assert count_params(RUN) == 125_264_640

def test_DEV_params():
    assert count_params(DEV) > 5_000_000 and count_params(DEV) < 15_000_000

def test_INVALIDE_params():
    with pytest.raises(ValueError):
        ModelConfig(
            n_heads=12,
            n_kv_heads=5
        )