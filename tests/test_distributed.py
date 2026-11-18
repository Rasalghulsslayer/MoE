import subprocess
import sys


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