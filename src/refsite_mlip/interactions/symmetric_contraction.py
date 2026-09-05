"""Standalone full-path factorized symmetric angular contraction.

This module consumes the fixed generalized real-CG basis from
``symmetric_cg``.  Standalone instances may own their basis buffers; residual
layers use the weights-only construction and receive the single externally
owned basis bank explicitly at each forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from torch import nn

from refsite_mlip.compatibility import import_e3nn_0_4_4

from .symmetric_cg import (
    GeneralizedCGCoefficients,
    SymmetricCGBasisBank,
    generate_generalized_cg,
)


_SUPPORTED_DTYPES = (torch.float32, torch.float64)


class SymmetricContractionError(ValueError):
    """Structured layout, basis, or runtime contraction failure."""

    def __init__(self, reason_code: str, message: str, *, field: str) -> None:
        self.reason_code = reason_code
        self.message = message
        self.field = field
        super().__init__(f"[{reason_code}] field={field!r} {message}")


def _error(
    reason_code: str, message: str, *, field: str
) -> SymmetricContractionError:
    return SymmetricContractionError(reason_code, message, field=field)


@dataclass(frozen=True)
class SymmetricContractionPathCount:
    order: int
    output_irrep: str
    path_count: int


@dataclass(frozen=True)
class SymmetricContractionDiagnostics:
    correlation_order: int
    output_irreps: str
    path_counts: tuple[SymmetricContractionPathCount, ...]
    parameter_count: int
    buffer_byte_count: int
    basis_kind: str
    dense_A_outer_materialized: bool
    horner_intermediate_shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SymmetricContractionResult:
    output: torch.Tensor
    correlation_order: int
    output_irreps: str
    order_contributions: tuple[torch.Tensor, ...] | None
    diagnostics: SymmetricContractionDiagnostics


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _error(
            "INVALID_INTEGER",
            "value must be a positive integer; bool is forbidden",
            field=field,
        )
    result = int(value)
    if result <= 0:
        raise _error(
            "INVALID_INTEGER", "value must be positive", field=field
        )
    return result


def _correlation_order(value: Any) -> int:
    result = _positive_integer(value, field="correlation_order")
    if result not in (1, 2, 3):
        raise _error(
            "UNSUPPORTED_CORRELATION_ORDER",
            "correlation order must be 1, 2, or 3",
            field="correlation_order",
        )
    return result


def _parse_input_layout(value: Any):
    _, o3 = import_e3nn_0_4_4()
    try:
        irreps = o3.Irreps(value)
    except Exception as error:
        raise _error(
            "INVALID_INPUT_IRREPS",
            f"failed to parse input irreps: {error}",
            field="input_irreps",
        ) from error
    if len(irreps) == 0 or irreps.dim == 0:
        raise _error(
            "INVALID_INPUT_IRREPS",
            "input irreps cannot be empty",
            field="input_irreps",
        )
    multiplicities = tuple(multiplicity for multiplicity, _ in irreps)
    if any(value <= 0 for value in multiplicities) or len(set(multiplicities)) != 1:
        raise _error(
            "NONUNIFORM_IRREP_MULTIPLICITY",
            "every input angular irrep must have one uniform positive multiplicity",
            field="input_irreps",
        )
    labels = tuple(str(irrep) for _, irrep in irreps)
    if len(set(labels)) != len(labels):
        raise _error(
            "DUPLICATE_INPUT_IRREP",
            "duplicate input irrep blocks are unsupported",
            field="input_irreps",
        )
    if max(irrep.l for _, irrep in irreps) > 2:
        raise _error(
            "UNSUPPORTED_ANGULAR_MOMENTUM",
            "standalone symmetric contraction supports input lmax <= 2",
            field="input_irreps",
        )
    if any(irrep.p != (-1) ** irrep.l for _, irrep in irreps):
        raise _error(
            "UNSUPPORTED_PARITY_LAYOUT",
            "input angular blocks must use spherical-harmonic parity (-1)^l",
            field="input_irreps",
        )
    channels = multiplicities[0]
    coupling_irreps = o3.Irreps([(1, irrep) for _, irrep in irreps])
    return irreps, coupling_irreps, channels


def _parse_output_layout(value: Any):
    _, o3 = import_e3nn_0_4_4()
    try:
        irreps = o3.Irreps(value)
    except Exception as error:
        raise _error(
            "INVALID_OUTPUT_IRREPS",
            f"failed to parse output irreps: {error}",
            field="output_irreps",
        ) from error
    if len(irreps) == 0 or irreps.dim == 0:
        raise _error(
            "INVALID_OUTPUT_IRREPS",
            "output irreps cannot be empty",
            field="output_irreps",
        )
    if any(multiplicity != 1 for multiplicity, _ in irreps):
        raise _error(
            "UNSUPPORTED_OUTPUT_MULTIPLICITY",
            "requested output angular irreps must be multiplicity-free",
            field="output_irreps",
        )
    labels = tuple(str(irrep) for _, irrep in irreps)
    if len(set(labels)) != len(labels):
        raise _error(
            "DUPLICATE_OUTPUT_IRREP",
            "duplicate output irrep blocks are unsupported",
            field="output_irreps",
        )
    if max(irrep.l for _, irrep in irreps) > 2:
        raise _error(
            "UNSUPPORTED_ANGULAR_MOMENTUM",
            "standalone symmetric contraction supports output lmax <= 2",
            field="output_irreps",
        )
    if any(irrep.p != (-1) ** irrep.l for _, irrep in irreps):
        raise _error(
            "UNSUPPORTED_PARITY_LAYOUT",
            "output angular blocks must use spherical-harmonic parity (-1)^l",
            field="output_irreps",
        )
    return irreps


def _validate_basis(
    basis: GeneralizedCGCoefficients,
    *,
    order: int,
    output_irrep: str,
    angular_dimension: int,
) -> tuple[torch.Tensor, int]:
    if not isinstance(basis, GeneralizedCGCoefficients):
        raise _error(
            "INVALID_CG_BASIS",
            "generalized-CG factory returned an invalid object",
            field="cg_basis",
        )
    if basis.correlation_order != order or basis.normalization != "component":
        raise _error(
            "MISMATCHED_CG_PATH",
            "generalized-CG order or normalization mismatch",
            field="cg_basis",
        )
    paths = basis.paths_for(output_irrep)
    if not paths or not all(path.metadata.nonzero for path in paths):
        raise _error(
            "MISSING_CG_PATH",
            "requested output/order has no complete nonzero full-path basis",
            field="cg_basis",
        )
    _, o3 = import_e3nn_0_4_4()
    try:
        output_dimension = o3.Irrep(output_irrep).dim
    except Exception as error:
        raise _error(
            "MISMATCHED_CG_PATH",
            f"invalid generalized-CG output irrep: {error}",
            field="cg_basis",
        ) from error
    expected_shape = (output_dimension,) + (angular_dimension,) * order
    for index, path in enumerate(paths):
        if path.metadata.path_index != index:
            raise _error(
                "MISMATCHED_CG_PATH",
                "generalized-CG path indices are not contiguous",
                field="cg_basis",
            )
        if path.metadata.coefficient_shape != expected_shape:
            raise _error(
                "MISMATCHED_CG_PATH",
                "generalized-CG coefficient shape is incompatible",
                field="cg_basis",
            )
    stacked = torch.stack(tuple(path.coefficient for path in paths), dim=0)
    if not bool(torch.all(torch.isfinite(stacked))):
        raise _error(
            "NONFINITE_CG_BASIS",
            "generalized-CG basis contains NaN or Infinity",
            field="cg_basis",
        )
    return stacked, len(paths)


class FactorizedSymmetricContraction(nn.Module):
    """MACE-style full-path Horner contraction for same-channel densities."""

    def __init__(
        self,
        input_irreps: Any,
        output_irreps: Any,
        *,
        correlation_order: int,
        central_dimension: int,
        normalization: str = "component",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        _external_basis_bank: SymmetricCGBasisBank | None = None,
    ) -> None:
        super().__init__()
        self.correlation_order = _correlation_order(correlation_order)
        self.central_dimension = _positive_integer(
            central_dimension, field="central_dimension"
        )
        if normalization != "component":
            raise _error(
                "UNSUPPORTED_NORMALIZATION",
                "only component normalization is supported",
                field="normalization",
            )
        if dtype not in _SUPPORTED_DTYPES:
            raise _error(
                "UNSUPPORTED_DTYPE",
                "dtype must be torch.float32 or torch.float64",
                field="dtype",
            )
        try:
            target_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise _error(
                "INVALID_DEVICE", f"invalid device: {error}", field="device"
            ) from error

        inputs, coupling_irreps, channels = _parse_input_layout(input_irreps)
        outputs = _parse_output_layout(output_irreps)
        self.input_irreps = inputs
        self.coupling_irreps = coupling_irreps
        self.requested_output_irreps = outputs
        self.channel_count = channels
        self.angular_dimension = coupling_irreps.dim
        _, o3 = import_e3nn_0_4_4()
        self.output_irreps = o3.Irreps(
            [(channels, irrep) for _, irrep in outputs]
        )
        self.normalization = normalization
        self._owns_basis_buffers = _external_basis_bank is None
        self._external_basis_fingerprint = (
            None
            if _external_basis_bank is None
            else _external_basis_bank.basis_fingerprint
        )
        if _external_basis_bank is not None:
            self._validate_external_bank_architecture(_external_basis_bank)

        basis_names: dict[tuple[int, int], str] = {}
        weight_names: dict[tuple[int, int], str] = {}
        path_counts: list[SymmetricContractionPathCount] = []
        for output_index, (_, output_irrep) in enumerate(outputs):
            output_label = str(output_irrep)
            output_bases: dict[int, tuple[torch.Tensor, int]] = {}
            for order in range(1, self.correlation_order + 1):
                if _external_basis_bank is None:
                    try:
                        basis = generate_generalized_cg(
                            coupling_irreps,
                            output_label,
                            order,
                            normalization=normalization,
                            canonical_dtype=torch.float64,
                        )
                    except Exception as error:
                        if isinstance(error, SymmetricContractionError):
                            raise
                        raise _error(
                            "CG_BASIS_GENERATION_FAILED",
                            f"generalized-CG generation failed: {error}",
                            field="cg_basis",
                        ) from error
                    stacked, path_count = _validate_basis(
                        basis,
                        order=order,
                        output_irrep=output_label,
                        angular_dimension=self.angular_dimension,
                    )
                else:
                    stacked = _external_basis_bank.basis_tensor(
                        order, output_label
                    )
                    path_count = int(stacked.shape[0])
                output_bases[order] = (stacked, path_count)
                if _external_basis_bank is None:
                    name = f"u_output_{output_index}_order_{order}"
                    self.register_buffer(
                        name,
                        stacked.to(device=target_device, dtype=dtype),
                        persistent=True,
                    )
                    basis_names[(output_index, order)] = name
                path_counts.append(
                    SymmetricContractionPathCount(
                        order=order,
                        output_irrep=output_label,
                        path_count=path_count,
                    )
                )

            # Match MACE full-path initialization: each order's iid normal
            # weights are divided by that order/output's number of paths.
            for order in range(self.correlation_order, 0, -1):
                path_count = output_bases[order][1]
                weight = torch.randn(
                    self.central_dimension,
                    path_count,
                    self.channel_count,
                    dtype=dtype,
                    device=target_device,
                ) / path_count
                name = f"weight_output_{output_index}_order_{order}"
                self.register_parameter(name, nn.Parameter(weight))
                weight_names[(output_index, order)] = name

        self._basis_names = tuple(sorted(basis_names.items()))
        self._weight_names = tuple(sorted(weight_names.items()))
        self._path_counts = tuple(path_counts)

    @classmethod
    def from_basis_bank(
        cls,
        input_irreps: Any,
        *,
        central_dimension: int,
        basis_bank: SymmetricCGBasisBank,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "FactorizedSymmetricContraction":
        if not isinstance(basis_bank, SymmetricCGBasisBank):
            raise _error(
                "INVALID_CG_BASIS_BANK",
                "basis_bank must be a SymmetricCGBasisBank",
                field="basis_bank",
            )
        return cls(
            input_irreps,
            basis_bank.requested_output_irreps,
            correlation_order=basis_bank.correlation_order,
            central_dimension=central_dimension,
            normalization=basis_bank.normalization,
            dtype=dtype,
            device=device,
            _external_basis_bank=basis_bank,
        )

    def _validate_external_bank_architecture(
        self, basis_bank: SymmetricCGBasisBank
    ) -> None:
        if not isinstance(basis_bank, SymmetricCGBasisBank):
            raise _error(
                "INVALID_CG_BASIS_BANK",
                "external basis must be a SymmetricCGBasisBank",
                field="basis_bank",
            )
        if (
            basis_bank.input_irreps != str(self.coupling_irreps)
            or basis_bank.requested_output_irreps
            != str(self.requested_output_irreps)
            or basis_bank.correlation_order != self.correlation_order
            or basis_bank.normalization != self.normalization
        ):
            raise _error(
                "MISMATCHED_CG_BASIS_BANK",
                "external basis architecture differs from this contraction",
                field="basis_bank",
            )
        if (
            self._external_basis_fingerprint is not None
            and basis_bank.basis_fingerprint
            != self._external_basis_fingerprint
        ):
            raise _error(
                "MISMATCHED_CG_BASIS_BANK",
                "external basis fingerprint differs from construction",
                field="basis_bank",
            )

    def _name(self, entries, output_index: int, order: int) -> str:
        key = (output_index, order)
        for candidate, name in entries:
            if candidate == key:
                return name
        raise _error(
            "MISSING_CG_PATH",
            "internal output/order state is missing",
            field="state",
        )

    def basis_tensor(self, output_index: int, order: int) -> torch.Tensor:
        if not self._owns_basis_buffers:
            raise _error(
                "EXTERNAL_CG_BASIS_REQUIRED",
                "weights-only contraction does not own generalized-CG buffers",
                field="basis_bank",
            )
        name = self._name(self._basis_names, output_index, order)
        value = self._buffers.get(name)
        if not isinstance(value, torch.Tensor):
            raise _error(
                "MISSING_CG_PATH",
                "persistent generalized-CG buffer is missing",
                field=name,
            )
        return value

    def weight_parameter(self, output_index: int, order: int) -> nn.Parameter:
        name = self._name(self._weight_names, output_index, order)
        value = self._parameters.get(name)
        if not isinstance(value, nn.Parameter):
            raise _error(
                "MISSING_WEIGHT_PARAMETER",
                "symmetric contraction weight is missing",
                field=name,
            )
        return value

    def _expected_basis_shape(self, output_index: int, order: int):
        output_irrep = self.requested_output_irreps[output_index].ir
        path_count = next(
            item.path_count
            for item in self._path_counts
            if item.output_irrep == str(output_irrep) and item.order == order
        )
        return (path_count, output_irrep.dim) + (
            self.angular_dimension,
        ) * order

    def _validated_state(
        self,
        output_index: int,
        order: int,
        basis_bank: SymmetricCGBasisBank | None,
    ):
        if self._owns_basis_buffers:
            if basis_bank is not None:
                raise _error(
                    "UNEXPECTED_CG_BASIS_BANK",
                    "internally-owned contraction must not receive an external basis",
                    field="basis_bank",
                )
            basis = self.basis_tensor(output_index, order)
        else:
            if basis_bank is None:
                raise _error(
                    "EXTERNAL_CG_BASIS_REQUIRED",
                    "weights-only contraction requires its external basis bank",
                    field="basis_bank",
                )
            self._validate_external_bank_architecture(basis_bank)
            output_irrep = str(self.requested_output_irreps[output_index].ir)
            basis = basis_bank.basis_tensor(order, output_irrep)
        weight = self.weight_parameter(output_index, order)
        basis_field = (
            self._name(self._basis_names, output_index, order)
            if self._owns_basis_buffers
            else f"basis_bank.order_{order}.output_{output_index}"
        )
        if tuple(basis.shape) != self._expected_basis_shape(output_index, order):
            raise _error(
                "MISMATCHED_CG_PATH",
                "persistent generalized-CG buffer shape is invalid",
                field=basis_field,
            )
        expected_weight = (
            self.central_dimension,
            basis.shape[0],
            self.channel_count,
        )
        if tuple(weight.shape) != expected_weight:
            raise _error(
                "MISMATCHED_WEIGHT_PARAMETER",
                "weight shape does not match central/path/channel layout",
                field=self._name(self._weight_names, output_index, order),
            )
        if basis.dtype != weight.dtype or basis.device != weight.device:
            raise _error(
                "CG_STATE_DTYPE_DEVICE_MISMATCH",
                "basis buffer and weight must share dtype and device",
                field=basis_field,
            )
        if not bool(torch.all(torch.isfinite(basis))):
            raise _error(
                "NONFINITE_CG_BASIS",
                "persistent generalized-CG buffer contains NaN or Infinity",
                field=basis_field,
            )
        if not bool(torch.all(torch.isfinite(weight))):
            raise _error(
                "NONFINITE_WEIGHT_PARAMETER",
                "symmetric contraction weight contains NaN or Infinity",
                field=self._name(self._weight_names, output_index, order),
            )
        return basis, weight

    def _pack_density(self, density: torch.Tensor) -> torch.Tensor:
        pieces = []
        start = 0
        for multiplicity, irrep in self.input_irreps:
            stop = start + multiplicity * irrep.dim
            pieces.append(
                density[:, start:stop].reshape(
                    density.shape[0], self.channel_count, irrep.dim
                )
            )
            start = stop
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _start_order(
        central: torch.Tensor,
        weight: torch.Tensor,
        basis: torch.Tensor,
        density: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        if order == 1:
            return torch.einsum(
                "sq,qpk,poa,ska->sko", central, weight, basis, density
            )
        if order == 2:
            return torch.einsum(
                "sq,qpk,poab,skb->skoa", central, weight, basis, density
            )
        return torch.einsum(
            "sq,qpk,poabc,skc->skoab", central, weight, basis, density
        )

    @staticmethod
    def _central_weighted(
        central: torch.Tensor,
        weight: torch.Tensor,
        basis: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        if order == 1:
            return torch.einsum("sq,qpk,poa->skoa", central, weight, basis)
        if order == 2:
            return torch.einsum("sq,qpk,poab->skoab", central, weight, basis)
        raise AssertionError("order-three basis is fused with its first A contraction")

    @staticmethod
    def _contract_one_density(
        intermediate: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        if intermediate.ndim == 5:
            return torch.einsum("skoab,skb->skoa", intermediate, density)
        if intermediate.ndim == 4:
            return torch.einsum("skoa,ska->sko", intermediate, density)
        raise AssertionError("unexpected Horner intermediate rank")

    def _horner_output(
        self,
        output_index: int,
        central: torch.Tensor,
        density: torch.Tensor,
        basis_bank: SymmetricCGBasisBank | None,
    ) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
        top_basis, top_weight = self._validated_state(
            output_index, self.correlation_order, basis_bank
        )
        intermediate = self._start_order(
            central,
            top_weight,
            top_basis,
            density,
            self.correlation_order,
        )
        shapes = [] if intermediate.ndim == 3 else [tuple(intermediate.shape)]
        for order in range(self.correlation_order - 1, 0, -1):
            basis, weight = self._validated_state(
                output_index, order, basis_bank
            )
            weighted = self._central_weighted(central, weight, basis, order)
            if weighted.shape != intermediate.shape:
                raise _error(
                    "HORNER_LAYOUT_MISMATCH",
                    "adjacent symmetric-correlation orders do not align",
                    field="horner_intermediate",
                )
            intermediate = self._contract_one_density(
                intermediate + weighted, density
            )
            if intermediate.ndim != 3:
                shapes.append(tuple(intermediate.shape))
        return intermediate, tuple(shapes)

    def _pure_order_output(
        self,
        output_index: int,
        order: int,
        central: torch.Tensor,
        density: torch.Tensor,
        basis_bank: SymmetricCGBasisBank | None,
    ) -> torch.Tensor:
        basis, weight = self._validated_state(
            output_index, order, basis_bank
        )
        value = self._start_order(central, weight, basis, density, order)
        for _ in range(order - 1):
            value = self._contract_one_density(value, density)
        return value

    def _flatten_outputs(self, values: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            tuple(value.reshape(value.shape[0], -1) for value in values), dim=-1
        )

    def _diagnostics(
        self, intermediate_shapes: tuple[tuple[int, ...], ...]
    ) -> SymmetricContractionDiagnostics:
        return SymmetricContractionDiagnostics(
            correlation_order=self.correlation_order,
            output_irreps=str(self.output_irreps),
            path_counts=self._path_counts,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            buffer_byte_count=sum(
                buffer.numel() * buffer.element_size() for buffer in self.buffers()
            ),
            basis_kind="full_path",
            dense_A_outer_materialized=False,
            horner_intermediate_shapes=tuple(intermediate_shapes),
        )

    def forward(
        self,
        density: torch.Tensor,
        central: torch.Tensor,
        *,
        return_order_contributions: bool = False,
        basis_bank: SymmetricCGBasisBank | None = None,
    ) -> SymmetricContractionResult:
        if not isinstance(density, torch.Tensor) or density.ndim != 2:
            raise _error(
                "INVALID_DENSITY_SHAPE",
                "density must be a rank-two [S, K*D] tensor",
                field="density",
            )
        if not isinstance(central, torch.Tensor) or central.ndim != 2:
            raise _error(
                "INVALID_CENTRAL_SHAPE",
                "central state must be a rank-two [S, Q] tensor",
                field="central",
            )
        if density.shape != (density.shape[0], self.input_irreps.dim):
            raise _error(
                "INVALID_DENSITY_SHAPE",
                "density dimension does not match input irreps",
                field="density",
            )
        if central.shape != (density.shape[0], self.central_dimension):
            raise _error(
                "INVALID_CENTRAL_SHAPE",
                "central state does not match site and central dimensions",
                field="central",
            )
        first_parameter = next(self.parameters())
        if first_parameter.dtype not in _SUPPORTED_DTYPES:
            raise _error(
                "UNSUPPORTED_DTYPE",
                "module floating state must be torch.float32 or torch.float64",
                field="state",
            )
        if (
            density.dtype != first_parameter.dtype
            or central.dtype != first_parameter.dtype
        ):
            raise _error(
                "DTYPE_MISMATCH",
                "density, central state, and module parameters must share dtype",
                field="density,central",
            )
        if (
            density.device != first_parameter.device
            or central.device != first_parameter.device
        ):
            raise _error(
                "DEVICE_MISMATCH",
                "density, central state, and module parameters must share device",
                field="density,central",
            )
        if not bool(torch.all(torch.isfinite(density))) or not bool(
            torch.all(torch.isfinite(central))
        ):
            raise _error(
                "NONFINITE_INPUT",
                "density and central state must be finite",
                field="density,central",
            )
        if type(return_order_contributions) is not bool:
            raise _error(
                "INVALID_RETURN_OPTION",
                "return_order_contributions must be a bool",
                field="return_order_contributions",
            )

        packed = self._pack_density(density)
        output_blocks = []
        intermediate_shapes = []
        for output_index in range(len(self.requested_output_irreps)):
            block, shapes = self._horner_output(
                output_index, central, packed, basis_bank
            )
            output_blocks.append(block)
            intermediate_shapes.extend(shapes)
        output = self._flatten_outputs(output_blocks)

        contributions = None
        if return_order_contributions:
            contribution_values = []
            for order in range(1, self.correlation_order + 1):
                blocks = [
                    self._pure_order_output(
                        output_index,
                        order,
                        central,
                        packed,
                        basis_bank,
                    )
                    for output_index in range(len(self.requested_output_irreps))
                ]
                contribution_values.append(self._flatten_outputs(blocks))
            contributions = tuple(contribution_values)

        if not bool(torch.all(torch.isfinite(output))):
            raise _error(
                "NONFINITE_OUTPUT",
                "symmetric contraction produced NaN or Infinity",
                field="output",
            )
        return SymmetricContractionResult(
            output=output,
            correlation_order=self.correlation_order,
            output_irreps=str(self.output_irreps),
            order_contributions=contributions,
            diagnostics=self._diagnostics(tuple(intermediate_shapes)),
        )


__all__ = [
    "FactorizedSymmetricContraction",
    "SymmetricContractionDiagnostics",
    "SymmetricContractionError",
    "SymmetricContractionPathCount",
    "SymmetricContractionResult",
]
