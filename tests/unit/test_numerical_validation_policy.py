from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from refsite_mlip.features import (
    ProbabilityMultipoleConfig,
    build_probability_multipoles,
    effective_probability_validation_tolerances,
)
from refsite_mlip.transport import (
    TRAIN_FIXED,
    TrainSinkhornConfig,
    TransportSupportConfig,
    solve_atom_vacancy_ot,
)


def _float32_transport():
    distances = torch.tensor(
        [
            [0.20, 1.00, 1.60],
            [0.90, 0.25, 1.10],
            [1.20, 0.80, 0.30],
            [1.45, 1.35, 1.25],
        ],
        dtype=torch.float32,
    )
    support = TransportSupportConfig(
        kind="compact_c2", cutoff=2.0, switch_width=0.5, candidate_skin=0.2
    )
    result = solve_atom_vacancy_ot(
        distances.square() / (2.0 * 1.5**2),
        0.5,
        TRAIN_FIXED,
        "sinkhorn",
        TrainSinkhornConfig(iterations=256),
        support_config=support,
        atom_distances=distances,
    )
    return result, distances


def test_automatic_probability_tolerance_is_dtype_and_size_aware():
    result, distances = _float32_transport()
    numbers = torch.tensor([6, 41, 6], dtype=torch.long)
    automatic = effective_probability_validation_tolerances(
        result.P, None
    )
    strict = effective_probability_validation_tolerances(
        result.P, 1.0e-9
    )

    assert automatic["simplex"] > torch.finfo(torch.float32).eps
    assert automatic["species_count"] >= automatic["simplex"]
    assert strict == {
        "simplex": 1.0e-9,
        "species_count": 1.0e-9,
        "vacancy_mass": 1.0e-9,
    }
    build_probability_multipoles(
        result.P,
        result.q,
        numbers,
        torch.zeros((*distances.shape, 3), dtype=torch.float32),
        ProbabilityMultipoleConfig(species_vocabulary=(6, 41)),
    )
    with pytest.raises(ValueError, match="balanced probability-field"):
        build_probability_multipoles(
            result.P,
            result.q,
            numbers,
            torch.zeros((*distances.shape, 3), dtype=torch.float32),
            ProbabilityMultipoleConfig(
                species_vocabulary=(6, 41), probability_tolerance=1.0e-9
            ),
        )


def test_float64_default_and_explicit_probability_contract_round_trip():
    automatic = ProbabilityMultipoleConfig(species_vocabulary=(6, 41))
    explicit = replace(automatic, probability_tolerance=3.0e-10)
    assert automatic.probability_tolerance is None
    assert ProbabilityMultipoleConfig.from_dict(automatic.to_dict()) == automatic
    assert ProbabilityMultipoleConfig.from_dict(explicit.to_dict()) == explicit


def test_support_diagnostics_separate_switch_boundaries():
    result, distances = _float32_transport()
    diagnostics = result.support_diagnostics
    assert diagnostics is not None
    assert diagnostics.switch_on_boundary_gap == pytest.approx(
        float(torch.min(torch.abs(distances - 1.5)))
    )
    assert diagnostics.cutoff_boundary_gap == pytest.approx(
        float(torch.min(torch.abs(distances - 2.0)))
    )
    assert diagnostics.candidate_boundary_gap == pytest.approx(
        float(torch.min(torch.abs(distances - 2.2)))
    )
    assert diagnostics.to_dict()["switch_on_boundary_gap"] == (
        diagnostics.switch_on_boundary_gap
    )
