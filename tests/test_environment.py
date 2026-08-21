from __future__ import annotations

import pytest
import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4


def test_required_versions():
    e3nn, _ = import_e3nn_0_4_4()
    assert torch.__version__ == "2.6.0+cu118"
    assert e3nn.__version__ == "0.4.4"


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_e3nn_solid_harmonic_zero_and_double_backward(device):
    _, o3 = import_e3nn_0_4_4()
    vector = torch.zeros((2, 3), dtype=torch.float64, device=device, requires_grad=True)
    harmonics = o3.SphericalHarmonics(
        "0e + 1o + 2e",
        normalize=False,
        normalization="component",
        irreps_in="1o",
    ).to(device=device, dtype=torch.float64)
    value = harmonics(vector)
    torch.testing.assert_close(
        value[:, 0], torch.ones(2, dtype=torch.float64, device=device)
    )
    torch.testing.assert_close(
        value[:, 1:], torch.zeros_like(value[:, 1:]), atol=0.0, rtol=0.0
    )
    first = torch.autograd.grad(value.square().sum(), vector, create_graph=True)[0]
    second = torch.autograd.grad(first.square().sum(), vector)[0]
    assert torch.all(torch.isfinite(first))
    assert torch.all(torch.isfinite(second))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_phase_solver_preserves_dtype_device(dtype, device):
    from refsite_mlip.phase.newton import solve_training_phase

    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    cross = torch.ones(3, dtype=complex_dtype, device=device, requires_grad=True)
    modes = torch.eye(3, dtype=torch.long, device=device)
    weights = torch.ones(3, dtype=dtype, device=device)
    phase = torch.zeros(3, dtype=dtype, device=device)
    result = solve_training_phase(
        cross, modes, weights, phase, (1.0, 1.0), (1.0, 0.5)
    )
    assert result.phase.dtype == dtype
    assert result.phase.device.type == device
    result.objective.backward()
    assert cross.grad is not None
    assert torch.all(torch.isfinite(cross.grad))


def test_e3nn_compatibility_restores_safe_globals():
    class CallerOwnedSafeGlobal:
        pass

    torch.serialization.add_safe_globals([CallerOwnedSafeGlobal])
    before = set(torch.serialization.get_safe_globals())
    import_e3nn_0_4_4()
    after = set(torch.serialization.get_safe_globals())
    assert after == before
    assert slice not in after


def test_compatibility_import_order_is_process_independent(tmp_path):
    import os
    import subprocess
    import sys

    root = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = root
    scripts = [
        """
import torch
before=set(torch.serialization.get_safe_globals())
import refsite_mlip
from refsite_mlip.compatibility import import_e3nn_0_4_4
import_e3nn_0_4_4()
assert set(torch.serialization.get_safe_globals())==before
""",
        """
import torch
class CallerOwned: pass
torch.serialization.add_safe_globals([CallerOwned])
before=set(torch.serialization.get_safe_globals())
from refsite_mlip.compatibility import import_e3nn_0_4_4
import_e3nn_0_4_4()
assert set(torch.serialization.get_safe_globals())==before
""",
        """
import torch
torch.serialization.add_safe_globals([slice])
before=set(torch.serialization.get_safe_globals())
from refsite_mlip.compatibility import import_e3nn_0_4_4
import_e3nn_0_4_4()
assert set(torch.serialization.get_safe_globals())==before
assert slice in torch.serialization.get_safe_globals()
""",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
