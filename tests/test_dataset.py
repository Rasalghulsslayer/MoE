import pytest 
import torch
import numpy as np
from training.dataset import load_tokens, make_sequences, get_batch

T = 10

@pytest.fixture
def tokens():
    return np.arange(1, 32005, dtype='uint16')

@pytest.fixture
def data_folder(tmp_path, tokens):
    foldername = tmp_path / "data"
    foldername.mkdir()
    filename = foldername / "train_test.bin"
    np.array(tokens, dtype=np.uint16).tofile(filename)
    return foldername

@pytest.fixture
def rng():
    return np.random.default_rng(42)

def test_dataset_shape(data_folder, tokens):
    dataset = load_tokens(data_folder / "train_test.bin")
    sequenced_dataset = make_sequences(dataset, T)
    assert sequenced_dataset.shape == (len(tokens) // (T+1), T+1)

def test_dataset_contiguous(data_folder, tokens):
    dataset = load_tokens(data_folder / "train_test.bin")
    assert np.array_equal(make_sequences(dataset, T).reshape(1, -1)[0], tokens[:(T+1) * (len(tokens) // (T+1))])

def test_dataset_logic(data_folder):
    dataset = load_tokens(data_folder / "train_test.bin")
    sequences = make_sequences(dataset, T)
    n = len(sequences)
    x, y = sequences[:,:-1], sequences[:,1:]
    assert x.shape == (n, T) and y.shape == (n, T)
    assert np.array_equal(x[:, 1:], y[:, :-1])

@pytest.fixture
def sequences(data_folder):
    dataset = load_tokens(data_folder / "train_test.bin")
    return make_sequences(dataset, T)


def test_dataset_numpy_to_torch(tokens):
    assert isinstance(torch.from_numpy(tokens), torch.Tensor)

def test_dataset_get_batch(sequences):
    B = 10
    rng1 = np.random.default_rng(42)
    x1, _ = get_batch(sequences, B, rng1)
    rng2 = np.random.default_rng(42)
    x2, _ = get_batch(sequences, B, rng2)
    rng3 = np.random.default_rng(0)
    x3, _ = get_batch(sequences, B, rng3)
    assert torch.equal(x1, x2)
    assert not torch.equal(x1, x3)

def test_dataset_batch_shape(sequences, rng):
    B = 2
    x, y = get_batch(sequences, B, rng)
    assert x.shape == y.shape == (B, T)

def test_dataset_batch_logic(data_folder, rng):
    B = 10
    dataset = load_tokens(data_folder / "train_test.bin")
    sequences = make_sequences(dataset, T)
    x, y = get_batch(sequences, B, rng)
    assert torch.equal(x[:, 1:], y[:, :-1])
