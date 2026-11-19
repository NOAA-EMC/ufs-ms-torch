"""The model supports one PyTorch release and reaches PyTorch internals through one module."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
COMPAT = SRC / "mpas_pytorch" / "_torch_compat.py"
POLICY = SRC / "ufs_physics_common" / "compile_policy.py"


def test_import_refuses_another_torch_release():
    code = "import torch\ntorch.__version__ = '2.8.1'\nimport mpas_pytorch._torch_compat\n"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode != 0
    assert "RuntimeError" in done.stderr and "torch==2.8.0" in done.stderr, done.stderr[-2000:]


@pytest.mark.parametrize("version", ["2.8.1", "2.9.0+cu128"])
def test_exact_numerics_refuses_another_torch_release(version, monkeypatch):
    from mpas_pytorch import _torch_compat
    from ufs_physics_common import compile_policy

    monkeypatch.setattr(torch, "__version__", version)
    with pytest.raises(RuntimeError, match="torch==2.8.0"):
        _torch_compat.require_supported_torch()
    with pytest.raises(RuntimeError, match="torch==2.8.0"):
        compile_policy.install_exact_numerics()
    monkeypatch.setattr(torch, "__version__", "2.8.0+cpu")  # a local build of the supported release
    _torch_compat.require_supported_torch()


def test_level_helpers_follow_forward_ad_and_functorch():
    import torch.autograd.forward_ad as fwad
    from mpas_pytorch import _torch_compat as compat

    assert compat.forward_ad_level() == -1 and not compat.forward_ad_active() and compat.functorch_level() is None
    with fwad.dual_level():
        assert compat.forward_ad_level() >= 0 and compat.forward_ad_active()
    seen = []

    def probe(x):
        seen.append((compat.functorch_level(), compat.is_functorch_wrapped(x)))
        return x * 2.0

    torch.func.vmap(probe)(torch.ones(3))
    assert seen[0][0] is not None and seen[0][1]


@pytest.mark.parametrize("name", ["torch._C._functorch", "forward_ad._current_level", "eval_frame"])
def test_private_torch_names_stay_in_the_compat_module(name):
    found = [f"{path.relative_to(SRC)}:{number}"
             for path in sorted(SRC.rglob("*.py")) if path not in (COMPAT, POLICY)
             for number, line in enumerate(path.read_text().splitlines(), 1) if name in line]
    assert not found, found
