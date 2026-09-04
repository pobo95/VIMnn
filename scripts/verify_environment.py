"""Verify the pinned eager-mode environment without installing anything."""

from __future__ import annotations

import ase
import numpy
import torch
import yaml

import refsite_mlip
from refsite_mlip.compatibility import import_e3nn_0_4_4


def _major_minor(version: str) -> tuple[int, int]:
    fields = version.split(".")
    try:
        return int(fields[0]), int(fields[1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"could not parse dependency version {version!r}") from error


def main() -> None:
    before = set(torch.serialization.get_safe_globals())
    e3nn, o3 = import_e3nn_0_4_4()
    after = set(torch.serialization.get_safe_globals())
    if before != after:
        raise RuntimeError("scoped e3nn compatibility changed process safe globals")
    if torch.__version__ != "2.6.0+cu118":
        raise RuntimeError(f"expected torch 2.6.0+cu118, got {torch.__version__}")
    if e3nn.__version__ != "0.4.4":
        raise RuntimeError(f"expected e3nn 0.4.4, got {e3nn.__version__}")
    if not ((1, 24) <= _major_minor(numpy.__version__) < (3, 0)):
        raise RuntimeError(f"expected numpy>=1.24,<3, got {numpy.__version__}")
    if not ((3, 22) <= _major_minor(ase.__version__) < (4, 0)):
        raise RuntimeError(f"expected ase>=3.22,<4, got {ase.__version__}")
    if not ((6, 0) <= _major_minor(yaml.__version__) < (7, 0)):
        raise RuntimeError(f"expected PyYAML>=6.0,<7, got {yaml.__version__}")
    vector = torch.zeros((1, 3), dtype=torch.float64, requires_grad=True)
    harmonics = o3.SphericalHarmonics(
        "0e + 1o + 2e",
        normalize=False,
        normalization="component",
        irreps_in="1o",
    )
    value = harmonics(vector)
    first = torch.autograd.grad(value.square().sum(), vector, create_graph=True)[0]
    torch.autograd.grad(first.square().sum(), vector)
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"e3nn={e3nn.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"ase={ase.__version__}")
    print(f"pyyaml={yaml.__version__}")
    print(f"refsite_mlip={refsite_mlip.__version__}")
    print("safe_globals_restored=True")
    print("eager_double_backward=ok")


if __name__ == "__main__":
    main()
