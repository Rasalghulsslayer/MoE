import subprocess
import sys
from training.dataset import sample_indices


def test_solo_pas_de_mode_distribue():
    result = subprocess.run(
        [sys.executable, "-m", "training.loop"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "None" in result.stdout


def test_torchrun_deux_rangs_se_rejoignent():
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

def test_sitributed_sample():
    N = 10
    idx0 = sample_indices(N, rank=0, world_size=2, epoch=0)
    idx1 = sample_indices(N, rank=1, world_size=2, epoch=0)
    
    assert set(idx0) & set(idx1) == set()        # disjoints
    assert set(idx0) | set(idx1) == set(range(N)) # union complète
    assert len(idx0) + len(idx1) == N