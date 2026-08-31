"""Vectorized truncated OT-weighted probability-density multipoles."""

from __future__ import annotations

import math
from typing import Optional

import torch

from refsite_mlip.compatibility import import_e3nn_0_4_4

from .radial import compact_radial_basis
from .result import (
    ChannelMetadata,
    ProbabilityMultipoleConfig,
    ProbabilityMultipoleResult,
)
from .solid_harmonics import harmonic_slice, regular_solid_harmonics
from .species import species_probabilities


def effective_probability_validation_tolerances(
    P: torch.Tensor,
    configured_tolerance: float | None,
) -> dict[str, float]:
    """Return validation-only roundoff limits without changing feature arithmetic.

    An explicitly configured tolerance is authoritative.  The automatic float64
    contract remains 1e-7.  For float32, the bound follows the pairwise reduction
    depth of the atom and site reductions, plus final additions.  Solver
    residuals, non-finiteness, and the actual P/q/features are never altered by
    this helper.
    """

    if configured_tolerance is not None:
        value = float(configured_tolerance)
        return {"simplex": value, "species_count": value, "vacancy_mass": value}
    base = 1.0e-7
    if P.dtype == torch.float64:
        return {"simplex": base, "species_count": base, "vacancy_mass": base}
    if P.dtype != torch.float32:
        raise ValueError("probability validation supports float32 and float64")
    sites, atoms = P.shape
    global_reduction_depth = (
        math.ceil(math.log2(max(atoms, 2)))
        + math.ceil(math.log2(max(sites, 2)))
        + 2
    )
    epsilon = torch.finfo(P.dtype).eps
    local_depth = math.ceil(math.log2(max(atoms, 2))) + 3
    local = max(base, local_depth * epsilon)
    count = max(base, global_reduction_depth * epsilon)
    return {"simplex": local, "species_count": count, "vacancy_mass": local}


def _validate_inputs(
    P: torch.Tensor,
    q: torch.Tensor,
    atomic_numbers: torch.Tensor,
    displacements: torch.Tensor,
    config: ProbabilityMultipoleConfig,
    site_types: Optional[torch.Tensor],
) -> None:
    config.validate()
    if P.ndim != 2:
        raise ValueError("P must have shape [M,N]")
    sites, atoms = P.shape
    if q.shape != (sites,):
        raise ValueError("q must have shape [M]")
    if atomic_numbers.shape != (atoms,) or atomic_numbers.dtype != torch.long:
        raise ValueError("atomic_numbers must be long with shape [N]")
    if displacements.shape != (sites, atoms, 3):
        raise ValueError("displacements must have shape [M,N,3]")
    for value in (P, q, displacements):
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError("feature floating inputs must use float32 or float64")
        if value.dtype != P.dtype or value.device != P.device:
            raise ValueError("P, q, and displacements must share dtype/device")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError("feature input contains NaN or Inf")
    if atomic_numbers.device != P.device:
        raise ValueError("atomic_numbers must share P device")
    if bool(torch.any(P < 0.0)) or bool(torch.any(q < 0.0)) or bool(
        torch.any(q > 1.0)
    ):
        raise ValueError("P and q must be nonnegative and q must not exceed one")
    if site_types is not None:
        if site_types.shape != (sites,) or site_types.dtype != torch.long:
            raise ValueError("site_types must be long with shape [M]")
        if site_types.device != P.device:
            raise ValueError("site_types must share P device")
        if config.site_type_vocabulary is None:
            raise ValueError(
                "site_types require a fixed site_type_vocabulary in config"
            )
        vocabulary = torch.tensor(
            config.site_type_vocabulary,
            dtype=torch.long,
            device=P.device,
        )
        known = site_types.unsqueeze(-1) == vocabulary.unsqueeze(0)
        if bool(torch.any(known.sum(dim=-1) != 1)):
            raise ValueError("unknown reference-site type")


def _layout(
    config: ProbabilityMultipoleConfig,
) -> tuple[object, tuple[ChannelMetadata, ...]]:
    _, o3 = import_e3nn_0_4_4()
    species_count = len(config.species_vocabulary)
    copies = species_count * config.n_radial
    irreps_out = o3.Irreps(
        f"{species_count}x0e + {copies}x0e + "
        f"{copies}x1o + {copies}x2e"
    )
    metadata = []
    offset = 0
    for species in config.species_vocabulary:
        metadata.append(
            ChannelMetadata(
                block_order=1,
                block_name="exact_species_occupancy",
                species=species,
                l=0,
                parity="e",
                radial_index=None,
                component_slice=(offset, offset + 1),
                exact_occupancy=True,
            )
        )
        offset += 1
    for block_order, l in enumerate(range(config.lmax + 1), start=2):
        dimension = 2 * l + 1
        parity = "e" if l % 2 == 0 else "o"
        for species in config.species_vocabulary:
            for radial in range(config.n_radial):
                metadata.append(
                    ChannelMetadata(
                        block_order=block_order,
                        block_name=f"compact_l{l}_multipole",
                        species=species,
                        l=l,
                        parity=parity,
                        radial_index=radial,
                        component_slice=(offset, offset + dimension),
                        exact_occupancy=False,
                    )
                )
                offset += dimension
    if offset != irreps_out.dim:
        raise RuntimeError("probability multipole layout dimension mismatch")
    return irreps_out, tuple(metadata)


def build_probability_multipoles(
    P: torch.Tensor,
    q: torch.Tensor,
    atomic_numbers: torch.Tensor,
    displacements: torch.Tensor,
    config: ProbabilityMultipoleConfig,
    site_types: Optional[torch.Tensor] = None,
) -> ProbabilityMultipoleResult:
    """Build features without solving OT or altering the supplied probabilities."""

    _validate_inputs(P, q, atomic_numbers, displacements, config, site_types)
    probabilities, indicator = species_probabilities(
        P, atomic_numbers, config.species_vocabulary
    )
    tolerances = effective_probability_validation_tolerances(
        P, config.probability_tolerance
    )
    # Validation reductions use float64 for float32 inputs, while the returned
    # probability/features retain the requested dtype and exact arithmetic.
    # This separates stored-plan error from an avoidable second reduction error.
    validation_dtype = torch.float64 if P.dtype == torch.float32 else P.dtype
    validation_P = P.to(validation_dtype)
    validation_q = q.to(validation_dtype)
    validation_indicator = indicator.to(validation_dtype)
    validation_probabilities = validation_P @ validation_indicator
    simplex_error = (
        validation_probabilities.sum(dim=1) + validation_q - 1.0
    ).abs().max()
    expected_counts = validation_indicator.sum(dim=0)
    count_error = (
        validation_probabilities.sum(dim=0) - expected_counts
    ).abs().max()
    vacancy_error = (
        validation_q.sum() - float(P.shape[0] - P.shape[1])
    ).abs()
    atom_column_error = (validation_P.sum(dim=0) - 1.0).abs().max()
    if (
        bool(simplex_error > tolerances["simplex"])
        or bool(count_error > tolerances["species_count"])
        or bool(vacancy_error > tolerances["vacancy_mass"])
        or bool(atom_column_error > tolerances["simplex"])
    ):
        raise ValueError(
            "P/q do not satisfy the balanced probability-field contract"
        )

    y = displacements / displacements.new_tensor(config.ell_feature)
    xi = torch.sum(y * y, dim=-1)
    radial = compact_radial_basis(
        xi,
        n_radial=config.n_radial,
        ell_feature=config.ell_feature,
        r_cut=config.r_cut,
    )
    harmonics, _ = regular_solid_harmonics(y, config.lmax)

    blocks = [probabilities]
    for l in range(config.lmax + 1):
        values = harmonics[..., harmonic_slice(l)]
        multipoles = torch.einsum(
            "si,ia,sin,sic->sanc",
            P,
            indicator,
            radial,
            values,
        )
        blocks.append(multipoles.reshape(P.shape[0], -1))
    equivariant = torch.cat(blocks, dim=1)
    irreps_out, metadata = _layout(config)
    raw_state = torch.cat((probabilities, q.unsqueeze(-1)), dim=1)
    return ProbabilityMultipoleResult(
        species_probabilities=probabilities,
        vacancy_probabilities=q,
        raw_probability_state=raw_state,
        equivariant_features=equivariant,
        irreps_out=irreps_out,
        channel_metadata=metadata,
        species_vocabulary=config.species_vocabulary,
        site_types=site_types,
        site_type_vocabulary=config.site_type_vocabulary,
        config_metadata=config.to_dict(),
    )
