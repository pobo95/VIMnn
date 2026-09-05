#!/usr/bin/env python
"""Opt-in numerical audit against the installed official MACE full-path code.

MACE is deliberately not a project dependency.  This script locates a separately
installed ``mace-torch`` distribution, loads only its official CG and symmetric
contraction source files, and compares them with refsite-mlip.  It does not
modify site-packages or global safe registrations.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.interactions.symmetric_contraction import (
    FactorizedSymmetricContraction,
)


ANGULAR = "0e + 1o + 2e"
ORDERS = (1, 2, 3)


def _load_source(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load official MACE source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _load_official_mace():
    distribution = importlib.metadata.distribution("mace-torch")
    version = distribution.version
    root = Path(distribution.locate_file("mace")).resolve()
    if version != "0.3.16":
        raise RuntimeError(
            f"this audit is pinned to official mace-torch 0.3.16, found {version}"
        )

    import_e3nn_0_4_4()
    mace_package = types.ModuleType("mace")
    mace_package.__path__ = [str(root)]
    tools_package = types.ModuleType("mace.tools")
    tools_package.__path__ = [str(root / "tools")]
    modules_package = types.ModuleType("mace.modules")
    modules_package.__path__ = [str(root / "modules")]
    sys.modules.update(
        {
            "mace": mace_package,
            "mace.tools": tools_package,
            "mace.modules": modules_package,
        }
    )
    cg = _load_source("mace.tools.cg", root / "tools" / "cg.py")
    _load_source("mace.tools.compile", root / "tools" / "compile.py")
    contraction = _load_source(
        "mace.modules.symmetric_contraction",
        root / "modules" / "symmetric_contraction.py",
    )
    return version, root, cg, contraction


def _canonical_mace_u(value: torch.Tensor, output_dimension: int, order: int):
    # MACE stores [M, a1, ..., anu, path], and squeezes M for scalar output.
    if output_dimension == 1 and value.ndim == order + 1:
        value = value.unsqueeze(0)
    return value.movedim(-1, 0).contiguous()


def _signed_path_mapping(ours: torch.Tensor, mace: torch.Tensor):
    if ours.shape != mace.shape:
        raise RuntimeError(f"incompatible U shapes: ours={ours.shape}, mace={mace.shape}")
    unused = set(range(mace.shape[0]))
    mapping = []
    maximum_error = 0.0
    for ours_index in range(ours.shape[0]):
        matches = []
        for mace_index in sorted(unused):
            for sign in (1, -1):
                error = float(
                    torch.max(torch.abs(ours[ours_index] - sign * mace[mace_index]))
                )
                if error <= 2.0e-14:
                    matches.append((error, mace_index, sign))
        if len(matches) != 1:
            raise RuntimeError(
                f"path {ours_index} has {len(matches)} signed-permutation matches"
            )
        error, mace_index, sign = matches[0]
        mapping.append((mace_index, sign))
        maximum_error = max(maximum_error, error)
        unused.remove(mace_index)
    if unused:
        raise RuntimeError(f"unmapped official MACE paths: {sorted(unused)}")
    return tuple(mapping), maximum_error


def _pack(module, density):
    pieces = []
    start = 0
    for multiplicity, irrep in module.input_irreps:
        stop = start + multiplicity * irrep.dim
        pieces.append(density[:, start:stop].reshape(density.shape[0], multiplicity, irrep.dim))
        start = stop
    return torch.cat(pieces, dim=-1)


def _unpack_gradient(module, packed):
    pieces = []
    start = 0
    for multiplicity, irrep in module.input_irreps:
        stop = start + irrep.dim
        pieces.append(packed[:, :, start:stop].reshape(packed.shape[0], multiplicity * irrep.dim))
        start = stop
    return torch.cat(pieces, dim=-1)


def _difference(left, right):
    absolute = torch.abs(left - right)
    maximum_absolute = torch.max(absolute)
    scale = torch.maximum(torch.max(torch.abs(left)), torch.max(torch.abs(right)))
    relative = maximum_absolute / torch.clamp(scale, min=torch.finfo(left.dtype).tiny)
    return float(maximum_absolute), float(relative)


def main() -> int:
    version, source_root, mace_cg, mace_contraction = _load_official_mace()
    _, o3 = import_e3nn_0_4_4()
    channels = 2
    central_dimension = 3
    module = FactorizedSymmetricContraction(
        "2x0e + 2x1o + 2x2e",
        ANGULAR,
        correlation_order=3,
        central_dimension=central_dimension,
        dtype=torch.float64,
    )
    original_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        official = mace_contraction.SymmetricContraction(
            irreps_in=o3.Irreps("2x0e + 2x1o + 2x2e"),
            irreps_out=o3.Irreps("2x0e + 2x1o + 2x2e"),
            correlation=3,
            irrep_normalization="component",
            path_normalization="element",
            use_reduced_cg=False,
            internal_weights=True,
            shared_weights=True,
            num_elements=central_dimension,
        )
    finally:
        torch.set_default_dtype(original_default_dtype)

    mappings = {}
    u_maximum = 0.0
    with torch.no_grad():
        for output_index, (_, output_irrep) in enumerate(module.requested_output_irreps):
            official_contraction = official.contractions[output_index]
            for order in ORDERS:
                ours_u = module.basis_tensor(output_index, order)
                official_u = _canonical_mace_u(
                    official_contraction.U_tensors(order), output_irrep.dim, order
                )
                mapping, error = _signed_path_mapping(ours_u, official_u)
                mappings[(output_index, order)] = mapping
                u_maximum = max(u_maximum, error)

                weight = module.weight_parameter(output_index, order)
                values = torch.sin(
                    torch.arange(weight.numel(), dtype=torch.float64).reshape_as(weight)
                    * 0.17
                    + output_index
                    + order
                )
                weight.copy_(values)
                target = (
                    official_contraction.weights_max
                    if order == 3
                    else official_contraction.weights[2 - order]
                )
                target.zero_()
                for ours_index, (mace_index, sign) in enumerate(mapping):
                    target[:, mace_index, :].copy_(sign * values[:, ours_index, :])

    density = torch.linspace(-0.31, 0.43, 54, dtype=torch.float64).reshape(3, 18)
    central = torch.tensor(
        [[0.7, -0.4, 0.2], [-0.1, 0.8, 0.5], [0.3, 0.1, -0.6]],
        dtype=torch.float64,
    )
    density_ours = density.clone().requires_grad_()
    density_mace = _pack(module, density).clone().requires_grad_()
    central_ours = central.clone().requires_grad_()
    central_mace = central.clone().requires_grad_()
    ours_result = module(
        density_ours, central_ours, return_order_contributions=True
    )
    mace_output = official(density_mace, central_mace)
    forward_absolute, forward_relative = _difference(ours_result.output, mace_output)

    probe = torch.linspace(-0.5, 0.7, ours_result.output.numel(), dtype=torch.float64).reshape_as(
        ours_result.output
    )
    ours_arguments = (density_ours, central_ours, *tuple(module.parameters()))
    mace_arguments = (density_mace, central_mace, *tuple(official.parameters()))
    ours_gradients = torch.autograd.grad(
        (ours_result.output * probe).sum(), ours_arguments
    )
    mace_gradients = torch.autograd.grad((mace_output * probe).sum(), mace_arguments)
    density_gradient_error = _difference(
        ours_gradients[0], _unpack_gradient(module, mace_gradients[0])
    )
    central_gradient_error = _difference(ours_gradients[1], mace_gradients[1])

    # Parameter iteration orders differ, so compare through semantic output/order keys.
    weight_errors = []
    ours_gradient_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(tuple(module.parameters()), ours_gradients[2:])
    }
    mace_gradient_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(tuple(official.parameters()), mace_gradients[2:])
    }
    for output_index in range(len(module.requested_output_irreps)):
        contraction = official.contractions[output_index]
        for order in ORDERS:
            ours_parameter = module.weight_parameter(output_index, order)
            mace_parameter = (
                contraction.weights_max if order == 3 else contraction.weights[2 - order]
            )
            mapping = mappings[(output_index, order)]
            canonical = torch.empty_like(ours_parameter)
            for ours_index, (mace_index, sign) in enumerate(mapping):
                canonical[:, ours_index, :] = (
                    sign * mace_gradient_by_id[id(mace_parameter)][:, mace_index, :]
                )
            weight_errors.append(
                _difference(ours_gradient_by_id[id(ours_parameter)], canonical)
            )

    rotation = o3.angles_to_matrix(
        torch.tensor(0.31, dtype=torch.float64),
        torch.tensor(0.47, dtype=torch.float64),
        torch.tensor(-0.23, dtype=torch.float64),
    )
    equivariance_errors = []
    for matrix in (rotation, -torch.eye(3, dtype=torch.float64)):
        transformed_density = density @ module.input_irreps.D_from_matrix(matrix).T
        expected_output = (
            module(density, central).output
            @ module.output_irreps.D_from_matrix(matrix).T
        )
        transformed_output = module(transformed_density, central).output
        equivariance_errors.append(_difference(transformed_output, expected_output))

    # Pure-order parity uses the exact same official full-path module with other
    # order weights temporarily zeroed; it does not substitute a reduced basis.
    order_errors = []
    original_weights = [parameter.detach().clone() for parameter in official.parameters()]
    ours_contributions = ours_result.order_contributions
    assert ours_contributions is not None
    for active_order in ORDERS:
        with torch.no_grad():
            for output_index, contraction in enumerate(official.contractions):
                for order in ORDERS:
                    parameter = (
                        contraction.weights_max
                        if order == 3
                        else contraction.weights[2 - order]
                    )
                    if order != active_order:
                        parameter.zero_()
        order_errors.append(_difference(ours_contributions[active_order - 1], official(_pack(module, density), central)))
        with torch.no_grad():
            for parameter, snapshot in zip(official.parameters(), original_weights):
                parameter.copy_(snapshot)

    report = {
        "mace": {
            "version": version,
            "release": f"mace-torch-{version}",
            "source": str(source_root),
            "use_reduced_cg": False,
            "use_cueq_cg": False,
            "normalization": "component",
        },
        "versions": {
            "torch": torch.__version__,
            "e3nn": importlib.metadata.version("e3nn"),
        },
        "path_counts": [
            {
                "order": item.order,
                "output_irrep": item.output_irrep,
                "count": item.path_count,
            }
            for item in module._path_counts
        ],
        "standalone": {
            "parameter_elements": sum(value.numel() for value in module.parameters()),
            "parameter_bytes": sum(
                value.numel() * value.element_size() for value in module.parameters()
            ),
            "buffer_elements": sum(value.numel() for value in module.buffers()),
            "buffer_bytes": sum(
                value.numel() * value.element_size() for value in module.buffers()
            ),
            "horner_intermediate_shapes": [
                list(shape) for shape in ours_result.diagnostics.horner_intermediate_shapes
            ],
            "dense_A_outer_materialized": False,
        },
        "mapping": {
            f"output_{output}_order_{order}": [
                {"ours": index, "mace": mace_index, "sign": sign}
                for index, (mace_index, sign) in enumerate(mapping)
            ]
            for (output, order), mapping in mappings.items()
        },
        "maximum_errors": {
            "u_absolute": u_maximum,
            "total_forward_absolute": forward_absolute,
            "total_forward_relative": forward_relative,
            "order_forward_absolute": max(value[0] for value in order_errors),
            "order_forward_relative": max(value[1] for value in order_errors),
            "density_gradient_absolute": density_gradient_error[0],
            "density_gradient_relative": density_gradient_error[1],
            "central_gradient_absolute": central_gradient_error[0],
            "central_gradient_relative": central_gradient_error[1],
            "weight_gradient_absolute": max(value[0] for value in weight_errors),
            "weight_gradient_relative": max(value[1] for value in weight_errors),
            "o3_equivariance_absolute": max(
                value[0] for value in equivariance_errors
            ),
            "o3_equivariance_relative": max(
                value[1] for value in equivariance_errors
            ),
        },
    }
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
