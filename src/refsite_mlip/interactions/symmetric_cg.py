"""Multiplicity-free generalized real Clebsch--Gordan coefficients.

This module independently implements the recursive generalized-CG formula used
by symmetric polynomial correlations.  It intentionally does not participate
in the current Potential execution path.  Coefficients use e3nn 0.4.4's real
spherical-harmonic convention and component normalization.

The canonical coefficient orientation is ``[m_out, a_1, ..., a_nu]``.  Every
``a`` axis spans the complete, multiplicity-free input angular basis.  Channel
multiplicity, sites, edges, and model state are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from numbers import Integral
from typing import Any

import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4


_SUPPORTED_ORDERS = (1, 2, 3)
_SUPPORTED_NORMALIZATIONS = ("component",)
_SUPPORTED_MATERIALIZATION_DTYPES = (torch.float32, torch.float64)


class SymmetricCGError(ValueError):
    """Structured generalized-CG construction or validation failure."""

    def __init__(self, reason_code: str, message: str, *, field: str) -> None:
        self.reason_code = reason_code
        self.message = message
        self.field = field
        super().__init__(f"[{reason_code}] field={field!r} {message}")


def _error(reason_code: str, message: str, *, field: str) -> SymmetricCGError:
    return SymmetricCGError(reason_code, message, field=field)


@dataclass(frozen=True)
class AngularIrrepBlock:
    """One multiplicity-free irrep and its slice in the angular input axis."""

    irrep: str
    start: int
    stop: int

    @property
    def dimension(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class GeneralizedCGPathMetadata:
    """Plain immutable identity and layout metadata for one coupling path."""

    correlation_order: int
    output_irrep: str
    path_index: int
    input_irreps: tuple[str, ...]
    intermediate_irreps: tuple[str, ...]
    coefficient_shape: tuple[int, ...]
    nonzero: bool


class GeneralizedCGPath:
    """Owned canonical coefficient tensor paired with immutable metadata.

    The constructor snapshots the supplied tensor.  ``coefficient`` and
    ``materialize`` return fresh tensors so callers cannot mutate the canonical
    CPU float64 storage held by this object.
    """

    __slots__ = ("_metadata", "_coefficient")

    def __init__(
        self,
        metadata: GeneralizedCGPathMetadata,
        coefficient: torch.Tensor,
    ) -> None:
        if not isinstance(metadata, GeneralizedCGPathMetadata):
            raise TypeError("metadata must be GeneralizedCGPathMetadata")
        if not isinstance(coefficient, torch.Tensor):
            raise TypeError("coefficient must be a torch.Tensor")
        if coefficient.device.type != "cpu" or coefficient.dtype != torch.float64:
            raise _error(
                "INVALID_CANONICAL_COEFFICIENT",
                "canonical coefficient must be a CPU float64 tensor",
                field="coefficient",
            )
        if tuple(coefficient.shape) != metadata.coefficient_shape:
            raise _error(
                "COEFFICIENT_SHAPE_MISMATCH",
                "coefficient shape differs from immutable path metadata",
                field="coefficient",
            )
        if coefficient.requires_grad:
            raise _error(
                "INVALID_CANONICAL_COEFFICIENT",
                "canonical coefficient must not require gradients",
                field="coefficient",
            )
        if not bool(torch.all(torch.isfinite(coefficient))):
            raise _error(
                "NONFINITE_COEFFICIENT",
                "generalized-CG coefficient contains NaN or Infinity",
                field="coefficient",
            )
        observed_nonzero = bool(torch.count_nonzero(coefficient))
        if observed_nonzero != metadata.nonzero:
            raise _error(
                "COEFFICIENT_ZERO_STATUS_MISMATCH",
                "coefficient zero status differs from immutable path metadata",
                field="metadata.nonzero",
            )
        self._metadata = metadata
        self._coefficient = coefficient.detach().clone().contiguous()

    @property
    def metadata(self) -> GeneralizedCGPathMetadata:
        return self._metadata

    @property
    def coefficient(self) -> torch.Tensor:
        """Return a caller-owned copy of the canonical CPU float64 tensor."""

        return self._coefficient.clone()

    @property
    def numel(self) -> int:
        return self._coefficient.numel()

    @property
    def nbytes(self) -> int:
        return self._coefficient.numel() * self._coefficient.element_size()

    def materialize(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return an explicit dtype/device copy without changing canonical data."""

        if dtype not in _SUPPORTED_MATERIALIZATION_DTYPES:
            raise _error(
                "UNSUPPORTED_MATERIALIZATION_DTYPE",
                "materialization dtype must be torch.float32 or torch.float64",
                field="dtype",
            )
        try:
            target = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise _error(
                "INVALID_MATERIALIZATION_DEVICE",
                f"invalid materialization device: {error}",
                field="device",
            ) from error
        return self._coefficient.to(device=target, dtype=dtype, copy=True)


@dataclass(frozen=True)
class GeneralizedCGOutputMetadata:
    """Path availability for one requested output irrep."""

    output_irrep: str
    output_index: int
    path_count: int
    nonzero_path_count: int

    @property
    def has_nonzero_path(self) -> bool:
        return self.nonzero_path_count > 0


@dataclass(frozen=True)
class GeneralizedCGCoefficients:
    """Canonical generalized-CG path collection."""

    correlation_order: int
    input_irreps: str
    requested_output_irreps: str
    normalization: str
    canonical_dtype: str
    input_dimension: int
    input_blocks: tuple[AngularIrrepBlock, ...]
    outputs: tuple[GeneralizedCGOutputMetadata, ...]
    paths: tuple[GeneralizedCGPath, ...]

    @property
    def path_count(self) -> int:
        return len(self.paths)

    @property
    def nonzero_path_count(self) -> int:
        return sum(path.metadata.nonzero for path in self.paths)

    @property
    def total_coefficient_bytes(self) -> int:
        return sum(path.nbytes for path in self.paths)

    def paths_for(self, output_irrep: str) -> tuple[GeneralizedCGPath, ...]:
        if type(output_irrep) is not str:
            raise TypeError("output_irrep must be a string")
        return tuple(
            path
            for path in self.paths
            if path.metadata.output_irrep == output_irrep
        )


def _validated_order(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _error(
            "INVALID_CORRELATION_ORDER",
            "correlation order must be an integer; bool is forbidden",
            field="correlation_order",
        )
    result = int(value)
    if result not in _SUPPORTED_ORDERS:
        raise _error(
            "UNSUPPORTED_CORRELATION_ORDER",
            "only correlation orders 1, 2, and 3 are supported",
            field="correlation_order",
        )
    return result


def _parse_multiplicity_free_irreps(value: Any, *, field: str):
    _, o3 = import_e3nn_0_4_4()
    if not isinstance(value, (str, o3.Irreps)):
        raise _error(
            "INVALID_IRREPS",
            "irreps must be an e3nn Irreps object or a nonempty string",
            field=field,
        )
    if isinstance(value, str) and not value.strip():
        raise _error("EMPTY_IRREPS", "irreps cannot be empty", field=field)
    try:
        irreps = o3.Irreps(str(value))
    except Exception as error:
        raise _error(
            "INVALID_IRREPS",
            f"failed to parse e3nn irreps: {error}",
            field=field,
        ) from error
    if len(irreps) == 0 or irreps.dim == 0:
        raise _error("EMPTY_IRREPS", "irreps cannot be empty", field=field)
    if any(multiplicity != 1 for multiplicity, _ in irreps):
        raise _error(
            "UNSUPPORTED_IRREP_MULTIPLICITY",
            "generalized-CG angular irreps must be multiplicity-free",
            field=field,
        )
    labels = tuple(str(irrep) for _, irrep in irreps)
    if len(set(labels)) != len(labels):
        raise _error(
            "UNSUPPORTED_IRREP_MULTIPLICITY",
            "duplicate angular irreps encode unsupported multiplicity",
            field=field,
        )
    return irreps


def _blocks(irreps) -> tuple[AngularIrrepBlock, ...]:
    result = []
    start = 0
    for multiplicity, irrep in irreps:
        if multiplicity != 1:  # Defensive; parsing already rejects this.
            raise AssertionError("internal multiplicity-free contract violated")
        stop = start + irrep.dim
        result.append(AngularIrrepBlock(str(irrep), start, stop))
        start = stop
    return tuple(result)


def _component_cg(o3, output_irrep, left_irrep, right_irrep) -> torch.Tensor:
    coefficient = o3.wigner_3j(
        output_irrep.l,
        left_irrep.l,
        right_irrep.l,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    coefficient = coefficient * math.sqrt(output_irrep.dim)
    if not bool(torch.all(torch.isfinite(coefficient))):
        raise _error(
            "NONFINITE_COEFFICIENT",
            "e3nn returned a nonfinite real Clebsch--Gordan coefficient",
            field="coefficient",
        )
    return coefficient


def _contains_irrep(product, requested) -> bool:
    return any(candidate == requested for candidate in product)


def _embed_local_coefficient(
    local: torch.Tensor,
    *,
    output_dimension: int,
    input_dimension: int,
    slices: tuple[slice, ...],
) -> torch.Tensor:
    shape = (output_dimension,) + (input_dimension,) * len(slices)
    coefficient = torch.zeros(shape, dtype=torch.float64, device="cpu")
    coefficient[(slice(None),) + slices] = local
    return coefficient


def generate_generalized_cg(
    input_irreps: Any,
    requested_output_irreps: Any,
    correlation_order: int,
    *,
    normalization: str = "component",
    canonical_dtype: torch.dtype = torch.float64,
) -> GeneralizedCGCoefficients:
    """Generate every multiplicity-free generalized real-CG coupling path.

    Path ordering is deterministic: requested output order, then the
    lexicographic tuple of input-irrep block indices, then increasing
    ``(L_intermediate, parity)`` for order three.  Intermediate angular momenta
    are never filtered by the requested final output irreps.
    """

    order = _validated_order(correlation_order)
    if normalization not in _SUPPORTED_NORMALIZATIONS:
        raise _error(
            "UNSUPPORTED_NORMALIZATION",
            "only e3nn component normalization is supported",
            field="normalization",
        )
    if canonical_dtype != torch.float64:
        raise _error(
            "UNSUPPORTED_CANONICAL_DTYPE",
            "canonical generalized-CG generation requires torch.float64",
            field="canonical_dtype",
        )

    _, o3 = import_e3nn_0_4_4()
    inputs = _parse_multiplicity_free_irreps(input_irreps, field="input_irreps")
    outputs = _parse_multiplicity_free_irreps(
        requested_output_irreps, field="requested_output_irreps"
    )
    input_blocks = _blocks(inputs)
    input_entries = tuple(irrep for _, irrep in inputs)
    input_slices = tuple(
        slice(block.start, block.stop) for block in input_blocks
    )
    input_dimension = inputs.dim

    paths: list[GeneralizedCGPath] = []
    output_metadata: list[GeneralizedCGOutputMetadata] = []
    input_index_tuples = tuple(
        itertools.product(range(len(input_entries)), repeat=order)
    )

    for output_index, (_, output_irrep) in enumerate(outputs):
        output_label = str(output_irrep)
        output_paths: list[GeneralizedCGPath] = []
        for input_indices in input_index_tuples:
            selected = tuple(input_entries[index] for index in input_indices)
            selected_slices = tuple(input_slices[index] for index in input_indices)
            local_paths: list[tuple[tuple[str, ...], torch.Tensor]] = []

            if order == 1:
                if selected[0] == output_irrep:
                    local_paths.append(
                        (
                            (),
                            torch.eye(
                                output_irrep.dim,
                                dtype=torch.float64,
                                device="cpu",
                            ),
                        )
                    )
            elif order == 2:
                if _contains_irrep(selected[0] * selected[1], output_irrep):
                    local_paths.append(
                        (
                            (),
                            _component_cg(
                                o3, output_irrep, selected[0], selected[1]
                            ),
                        )
                    )
            else:
                intermediate_irreps = sorted(
                    tuple(selected[0] * selected[1]),
                    key=lambda irrep: (irrep.l, irrep.p),
                )
                for intermediate in intermediate_irreps:
                    if not _contains_irrep(intermediate * selected[2], output_irrep):
                        continue
                    left = _component_cg(
                        o3, intermediate, selected[0], selected[1]
                    )
                    right = _component_cg(
                        o3, output_irrep, intermediate, selected[2]
                    )
                    local = torch.einsum("omc,mab->oabc", right, left)
                    local_paths.append(((str(intermediate),), local))

            for intermediate_labels, local in local_paths:
                coefficient = _embed_local_coefficient(
                    local,
                    output_dimension=output_irrep.dim,
                    input_dimension=input_dimension,
                    slices=selected_slices,
                )
                metadata = GeneralizedCGPathMetadata(
                    correlation_order=order,
                    output_irrep=output_label,
                    path_index=len(output_paths),
                    input_irreps=tuple(str(irrep) for irrep in selected),
                    intermediate_irreps=intermediate_labels,
                    coefficient_shape=tuple(coefficient.shape),
                    nonzero=bool(torch.count_nonzero(coefficient)),
                )
                output_paths.append(GeneralizedCGPath(metadata, coefficient))

        paths.extend(output_paths)
        output_metadata.append(
            GeneralizedCGOutputMetadata(
                output_irrep=output_label,
                output_index=output_index,
                path_count=len(output_paths),
                nonzero_path_count=sum(
                    path.metadata.nonzero for path in output_paths
                ),
            )
        )

    return GeneralizedCGCoefficients(
        correlation_order=order,
        input_irreps=str(inputs),
        requested_output_irreps=str(outputs),
        normalization=normalization,
        canonical_dtype="float64",
        input_dimension=input_dimension,
        input_blocks=input_blocks,
        outputs=tuple(output_metadata),
        paths=tuple(paths),
    )


__all__ = [
    "AngularIrrepBlock",
    "GeneralizedCGCoefficients",
    "GeneralizedCGOutputMetadata",
    "GeneralizedCGPath",
    "GeneralizedCGPathMetadata",
    "SymmetricCGError",
    "generate_generalized_cg",
]
