import subprocess
import re
import sys
import torch
from training.dataset import sample_indices

def test_distributed_solo():
    result = subprocess.run(
        [sys.executable, "-m", "training.loop"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "None" in result.stdout


def test_distributed_torchrun_communication():
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=2",
         "--master-addr=127.0.0.1", "--master-port=29517",
         "-m", "training.loop"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0
    assert "(0, 2)" in result.stdout
    assert "(1, 2)" in result.stdout

def test_distributed_sample():
    N = 10
    idx0 = sample_indices(N, rank=0, world_size=2, epoch=0)
    idx1 = sample_indices(N, rank=1, world_size=2, epoch=0)
    
    assert set(idx0) & set(idx1) == set()        # disjoints
    assert set(idx0) | set(idx1) == set(range(N)) # union complète
    assert len(idx0) + len(idx1) == N

def test_distributed_epoch_change():
    a = sample_indices(100, rank=0, world_size=2, epoch=0)
    b = sample_indices(100, rank=0, world_size=2, epoch=1)
    assert list(a) != list(b)

def test_distributed_smoke_batches_change():
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=2",
         "--master-addr=127.0.0.1", "--master-port=29518",
         "-m", "tests.smoke_dist_train"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    sums = re.findall(r"\[BATCH0\] rank (\d) : (\d+)", result.stdout)
    assert len(sums) == 2, result.stdout          
    assert sums[0][1] != sums[1][1], result.stdout   

def test_distributed_smoke_ddp():
    result_noddp = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=1",
         "--master-addr=127.0.0.1", "--master-port=29518",
         "-m", "tests.smoke_noddp_train"],
        capture_output=True, text=True, timeout=300
    )
    result_ddp = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         "--nproc_per_node=2",
         "--master-addr=127.0.0.1", "--master-port=29518",
         "-m", "test.smoke_ddp_train"],
         capture_output=True, text=True, timeout=300
    )
    assert result_noddp.returncode == 0 and result_ddp.returncode == 0
    solo_grad = torch.load("tests/smoke_results/not_dist_grads.pt")["grads"]
    dist_grad = torch.load("tests/smoke_results/dist_grads0.pt")["grads"]
    for a, b in zip(solo_grad, dist_grad): assert torch.allclose(a, b, atol=1e-6)
    solo_loss = torch.load("tests/smoke_results/not_dist_grads.pt")["loss"]
    dist_loss_0 = torch.load("tests/smoke_results/dist_grads0.pt")["loss"]
    dist_loss_1 = torch.load("tests/smoke_results/dist_grads1.pt")["loss"]
    assert torch.allclose(((dist_loss_0 + dist_loss_1) / 2), solo_loss, atol=1e-6)



