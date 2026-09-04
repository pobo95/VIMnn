from __future__ import annotations

from dataclasses import replace
import math

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
from refsite_mlip.transport.diagnostics import build_result
from refsite_mlip.transport.dual import transport_plan
from refsite_mlip.transport.problem import build_ot_problem


def _float32_transport(*, diagnostic_tolerance=None):
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
        TrainSinkhornConfig(
            iterations=256,
            diagnostic_tolerance=diagnostic_tolerance,
        ),
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
    requested_strict = effective_probability_validation_tolerances(
        result.P, 1.0e-9
    )

    assert automatic["simplex"] > torch.finfo(torch.float32).eps
    assert automatic["species_count"] >= automatic["simplex"]
    assert requested_strict == automatic
    build_probability_multipoles(
        result.P,
        result.q,
        numbers,
        torch.zeros((*distances.shape, 3), dtype=torch.float32),
        ProbabilityMultipoleConfig(species_vocabulary=(6, 41)),
    )
    # A configured check cannot be stricter than representable float32
    # reductions, but a genuinely unbalanced probability is still rejected.
    invalid_q = result.q.clone()
    invalid_q[0] += 20.0 * requested_strict["simplex"]
    with pytest.raises(ValueError, match="balanced probability-field"):
        build_probability_multipoles(
            result.P,
            invalid_q,
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


def test_train_fixed_result_is_exactly_reconstructible_from_returned_duals():
    seed = torch.tensor(
        [[0.4, 0.6], [0.6, 0.40000024]], dtype=torch.float32
    )
    cost = -torch.log(seed)
    problem = build_ot_problem(cost, 1.0)
    f = torch.zeros(2, dtype=torch.float32)
    g = torch.zeros(2, dtype=torch.float32)
    expected = transport_plan(problem, f, g)
    for converged, tolerance in ((False, 1.0e-12), (True, 1.0)):
        result = build_result(
            problem,
            f,
            g,
            converged=converged,
            sinkhorn_iterations=1,
            newton_iterations=0,
            cg_iterations=0,
            line_search_reductions=0,
            fallback_used=False,
            solver_name="fixture",
            path_name=TRAIN_FIXED,
            effective_diagnostic_tolerance=tolerance,
        )
        assert torch.equal(result.gamma, expected)
        assert torch.equal(result.P, expected[:, : problem.num_atoms])
        assert torch.equal(result.q, torch.zeros(2, dtype=torch.float32))
        assert torch.equal(result.f, f)
        assert torch.equal(result.g, g)


def test_train_fixed_diagnostic_tolerance_cannot_change_plan_or_duals():
    strict, _ = _float32_transport(diagnostic_tolerance=1.0e-12)
    loose, _ = _float32_transport(diagnostic_tolerance=1.0)
    assert not bool(strict.converged)
    assert bool(loose.converged)
    for name in ("gamma", "P", "q", "f", "g"):
        assert torch.equal(getattr(strict, name), getattr(loose, name))
    assert strict.effective_diagnostic_tolerance == 1.0e-12
    assert loose.effective_diagnostic_tolerance == 1.0


@pytest.mark.parametrize(
    ("P", "q"),
    (
        (
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            torch.tensor([-5.0e-8, 1.0], dtype=torch.float64),
        ),
        (
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            torch.tensor([1.0 + 5.0e-8, 0.0], dtype=torch.float64),
        ),
    ),
)
def test_vacancy_probability_bounds_allow_only_numerical_roundoff(P, q):
    result = build_probability_multipoles(
        P,
        q,
        torch.tensor([6], dtype=torch.long),
        torch.zeros((2, 1, 3), dtype=torch.float64),
        ProbabilityMultipoleConfig((6,)),
    )
    assert torch.equal(result.vacancy_probabilities, q)

    invalid = q.clone()
    invalid[0] = -1.0e-3 if q[0] <= 0.0 else 1.001
    with pytest.raises(ValueError, match="outside.*probability bounds"):
        build_probability_multipoles(
            P,
            invalid,
            torch.tensor([6], dtype=torch.long),
            torch.zeros((2, 1, 3), dtype=torch.float64),
            ProbabilityMultipoleConfig((6,)),
        )


def test_loose_dense_residual_tolerance_cannot_expand_q_physical_bounds():
    P = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    q = torch.tensor([-0.5, 1.5], dtype=torch.float64)
    config = ProbabilityMultipoleConfig(
        (6,), probability_tolerance=1.0
    )
    with pytest.raises(ValueError, match="outside.*probability bounds"):
        build_probability_multipoles(
            P,
            q,
            torch.tensor([6], dtype=torch.long),
            torch.zeros((2, 1, 3), dtype=torch.float64),
            config,
        )

    result, distances = _float32_transport()
    original_P = result.P.clone()
    original_q = result.q.clone()
    features = build_probability_multipoles(
        result.P,
        result.q,
        torch.tensor([6, 41, 6], dtype=torch.long),
        torch.zeros((*distances.shape, 3), dtype=torch.float32),
        ProbabilityMultipoleConfig((6, 41), probability_tolerance=1.0),
    )
    assert torch.equal(result.P, original_P)
    assert torch.equal(result.q, original_q)
    assert torch.equal(features.vacancy_probabilities, original_q)


def test_dense_float32_species_count_allows_column_roundoff_accumulation():
    sites = atoms = 64
    epsilon = torch.finfo(torch.float32).eps
    diagonal = torch.cat(
        (
            torch.full((32,), 1.0 + epsilon, dtype=torch.float32),
            torch.full((32,), 1.0 - epsilon, dtype=torch.float32),
        )
    )
    P = torch.diag(diagonal)
    q = torch.zeros(sites, dtype=torch.float32)
    numbers = torch.tensor([6] * 32 + [41] * 32, dtype=torch.long)
    displacements = torch.zeros((sites, atoms, 3), dtype=torch.float32)
    config = ProbabilityMultipoleConfig((6, 41))
    tolerances = effective_probability_validation_tolerances(P, None)

    assert tolerances["species_count"] >= (
        atoms
        + math.ceil(math.log2(atoms))
        + math.ceil(math.log2(sites))
        + 2
    ) * epsilon
    features = build_probability_multipoles(
        P, q, numbers, displacements, config
    )
    assert torch.equal(features.vacancy_probabilities, q)
    assert torch.equal(torch.diagonal(P), diagonal)

    invalid = P.clone()
    invalid[0, 0] += 20.0 * tolerances["species_count"]
    with pytest.raises(ValueError, match="balanced probability-field"):
        build_probability_multipoles(
            invalid, q, numbers, displacements, config
        )


def test_dense_aggregate_vacancy_supports_zero_atoms_without_empty_reduction():
    problem = build_ot_problem(torch.empty((3, 0), dtype=torch.float64), 0.5)
    zeros = torch.zeros(3, dtype=torch.float64)
    result = build_result(
        problem,
        zeros,
        torch.zeros(1, dtype=torch.float64),
        converged=True,
        sinkhorn_iterations=1,
        newton_iterations=0,
        cg_iterations=0,
        line_search_reductions=0,
        fallback_used=False,
        solver_name="fixture",
        path_name=TRAIN_FIXED,
    )
    assert result.P.shape == (3, 0)
    assert torch.equal(result.q, torch.ones(3, dtype=torch.float64))
    features = build_probability_multipoles(
        result.P,
        result.q,
        torch.empty(0, dtype=torch.long),
        torch.empty((3, 0, 3), dtype=torch.float64),
        ProbabilityMultipoleConfig((6, 41)),
    )
    assert torch.equal(
        features.species_probabilities,
        torch.zeros((3, 2), dtype=torch.float64),
    )


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
