from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from refsite_mlip.data import StructureSample
from refsite_mlip.training import (
    AtomicBaselineConfig,
    AtomicBaselineFit,
    fit_atomic_baseline,
)


def _sample(sample_id, composition, vocabulary, energy, *, dtype=torch.float64):
    atomic_numbers = []
    for species, count in zip(vocabulary, composition):
        atomic_numbers.extend([species] * count)
    num_atoms = len(atomic_numbers)
    return StructureSample(
        sample_id=sample_id,
        positions=torch.zeros((num_atoms, 3), dtype=dtype),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        cell=torch.eye(3, dtype=dtype),
        pbc=torch.ones(3, dtype=torch.bool),
        origin=torch.zeros(3, dtype=dtype),
        template_id="template",
        energy=None if energy is None else torch.tensor(energy, dtype=dtype),
    )


def _dataset(compositions, energies, vocabulary, *, dtype=torch.float64):
    return tuple(
        _sample(f"structure-{index}", row, vocabulary, energy, dtype=dtype)
        for index, (row, energy) in enumerate(zip(compositions, energies))
    )


def _oracle(compositions, energies, weighting):
    matrix = torch.tensor(compositions, dtype=torch.float64)
    target = torch.tensor(energies, dtype=torch.float64)
    if weighting == "per_atom":
        counts = matrix.sum(dim=1)
        matrix = matrix / counts[:, None]
        target = target / counts
    return torch.linalg.lstsq(matrix, target).solution


def test_full_rank_noiseless_exact_recovery_and_extensivity():
    vocabulary = (6, 41, 8)
    expected = torch.tensor([-1.25, 2.5, 0.75], dtype=torch.float64)
    compositions = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 1))
    energies = [float(torch.dot(torch.tensor(row, dtype=torch.float64), expected)) for row in compositions]
    dataset = _dataset(compositions, energies, vocabulary)
    fit = fit_atomic_baseline(
        dataset,
        range(len(dataset)),
        vocabulary,
        AtomicBaselineConfig(),
    )

    torch.testing.assert_close(fit.baseline_energies, expected, atol=2.0e-15, rtol=2.0e-15)
    assert fit.rank == 3 and not fit.rank_deficient
    assert fit.num_valid_energy_structures == 4
    assert fit.training_sample_ids == tuple(sample.sample_id for sample in dataset)
    assert fit.species_occurrence_counts.tolist() == [3, 4, 2]
    assert fit.baseline_energies.dtype == torch.float64
    assert fit.baseline_energies.device.type == "cpu"
    composition = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64)
    assert torch.equal(
        torch.dot(2.0 * composition, fit.baseline_energies),
        2.0 * torch.dot(composition, fit.baseline_energies),
    )


@pytest.mark.parametrize("weighting", ["per_structure", "per_atom"])
def test_noisy_weighting_matches_lstsq_oracle(weighting):
    vocabulary = (6, 41)
    compositions = ((1, 0), (0, 1), (2, 1), (1, 3), (4, 1))
    matrix = torch.tensor(compositions, dtype=torch.float64)
    expected = torch.tensor([-0.8, 1.7], dtype=torch.float64)
    noise = torch.tensor([0.04, -0.02, 0.03, -0.05, 0.01], dtype=torch.float64)
    energies = (matrix @ expected + noise).tolist()
    fit = fit_atomic_baseline(
        _dataset(compositions, energies, vocabulary, dtype=torch.float32),
        range(len(compositions)),
        vocabulary,
        AtomicBaselineConfig(weighting=weighting),
    )
    oracle = _oracle(compositions, energies, weighting)
    torch.testing.assert_close(fit.baseline_energies, oracle, atol=2.0e-7, rtol=2.0e-7)
    assert fit.residual_rmse > 0.0 and fit.residual_mae > 0.0
    assert fit.weighted_objective > 0.0


def test_ridge_solution_matches_normal_equation_oracle():
    vocabulary = (6, 41)
    compositions = ((1, 0), (0, 1), (2, 1), (1, 2))
    energies = (-1.0, 2.0, 0.3, 3.2)
    ridge = 0.4
    fit = fit_atomic_baseline(
        _dataset(compositions, energies, vocabulary),
        range(4),
        vocabulary,
        AtomicBaselineConfig(ridge=ridge),
    )
    matrix = torch.tensor(compositions, dtype=torch.float64)
    target = torch.tensor(energies, dtype=torch.float64)
    oracle = torch.linalg.solve(
        matrix.T @ matrix + ridge * torch.eye(2, dtype=torch.float64),
        matrix.T @ target,
    )
    torch.testing.assert_close(fit.baseline_energies, oracle, atol=2.0e-15, rtol=2.0e-15)
    residual = matrix @ oracle - target
    objective = residual.square().sum() + ridge * oracle.square().sum()
    assert fit.weighted_objective == pytest.approx(float(objective), abs=2.0e-15)


def test_missing_energy_is_excluded_and_nontraining_energy_cannot_leak():
    vocabulary = (6, 41)
    dataset = _dataset(
        ((1, 0), (0, 1), (1, 1), (4, 3)),
        (-1.0, 2.0, None, 5000.0),
        vocabulary,
    )
    config = AtomicBaselineConfig()
    first = fit_atomic_baseline(dataset, (0, 1, 2), vocabulary, config)
    changed_validation = tuple(dataset[:3]) + (
        replace(dataset[3], energy=torch.tensor(-9000.0, dtype=torch.float64)),
    )
    second = fit_atomic_baseline(
        changed_validation, (0, 1, 2), vocabulary, config
    )
    assert first.training_sample_ids == ("structure-0", "structure-1")
    assert first.num_valid_energy_structures == 2
    assert torch.equal(first.baseline_energies, second.baseline_energies)
    torch.testing.assert_close(
        first.baseline_energies,
        torch.tensor([-1.0, 2.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_species_order_unknown_and_absent_species_contracts():
    vocabulary = (41, 6)
    dataset = _dataset(((1, 0), (0, 1)), (3.0, -2.0), vocabulary)
    fit = fit_atomic_baseline(dataset, (0, 1), vocabulary, AtomicBaselineConfig())
    assert fit.species_vocabulary == vocabulary
    torch.testing.assert_close(
        fit.baseline_energies,
        torch.tensor([3.0, -2.0], dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )

    unknown = (_sample("unknown", (1,), (8,), 1.0),)
    with pytest.raises(ValueError, match="unknown species.*unknown"):
        fit_atomic_baseline(unknown, (0,), (6, 41), AtomicBaselineConfig())

    absent = (_sample("absent", (2, 0), (6, 41), -2.0),)
    with pytest.raises(ValueError, match="species absent.*41"):
        fit_atomic_baseline(absent, (0,), (6, 41), AtomicBaselineConfig())


def test_rank_deficient_default_failure_and_minimum_norm_diagnostics():
    vocabulary = (6, 41)
    dataset = _dataset(((1, 1), (2, 2)), (3.0, 6.0), vocabulary)
    with pytest.raises(ValueError, match="rank deficient.*minimum_norm"):
        fit_atomic_baseline(dataset, (0, 1), vocabulary, AtomicBaselineConfig())

    fit = fit_atomic_baseline(
        dataset,
        (0, 1),
        vocabulary,
        AtomicBaselineConfig(rank_policy="minimum_norm"),
    )
    torch.testing.assert_close(
        fit.baseline_energies,
        torch.tensor([1.5, 1.5], dtype=torch.float64),
        atol=2.0e-15,
        rtol=2.0e-15,
    )
    assert fit.rank == 1 and fit.rank_deficient
    assert math.isinf(fit.condition_number)
    assert fit.species_occurrence_counts.tolist() == [3, 3]


def test_index_and_empty_energy_validation():
    dataset = _dataset(((1,),), (None,), (6,))
    with pytest.raises(ValueError, match="duplicates"):
        fit_atomic_baseline(dataset, (0, 0), (6,), AtomicBaselineConfig())
    with pytest.raises(IndexError, match="out of range"):
        fit_atomic_baseline(dataset, (1,), (6,), AtomicBaselineConfig())
    with pytest.raises(IndexError, match="out of range"):
        fit_atomic_baseline(dataset, (-1,), (6,), AtomicBaselineConfig())
    with pytest.raises(TypeError, match="indices must be integers"):
        fit_atomic_baseline(dataset, (True,), (6,), AtomicBaselineConfig())
    with pytest.raises(ValueError, match="no valid energy"):
        fit_atomic_baseline(dataset, (0,), (6,), AtomicBaselineConfig())


def test_config_fit_serialization_and_deterministic_repeatability():
    config = AtomicBaselineConfig(
        weighting="per_atom", rcond=1.0e-12, ridge=0.1, rank_policy="error"
    )
    assert AtomicBaselineConfig.from_dict(config.to_dict()) == config
    with pytest.raises((TypeError, ValueError)):
        AtomicBaselineConfig(ridge=True)
    with pytest.raises(ValueError):
        AtomicBaselineConfig(rcond=float("nan"))
    with pytest.raises(ValueError):
        AtomicBaselineConfig(weighting="invalid")
    with pytest.raises(ValueError):
        AtomicBaselineConfig(rank_policy="invalid")

    vocabulary = (6, 41)
    dataset = _dataset(((1, 0), (0, 1), (2, 1)), (-1.0, 2.0, 0.2), vocabulary)
    first = fit_atomic_baseline(dataset, range(3), vocabulary, config)
    second = fit_atomic_baseline(dataset, range(3), vocabulary, config)
    assert torch.equal(first.baseline_energies, second.baseline_energies)
    assert torch.equal(first.singular_values, second.singular_values)
    assert first.to_dict() == second.to_dict()

    restored = AtomicBaselineFit.from_dict(first.to_dict())
    assert restored.to_dict() == first.to_dict()
    assert restored.baseline_energies.dtype == torch.float64
    assert restored.baseline_energies.device.type == "cpu"
