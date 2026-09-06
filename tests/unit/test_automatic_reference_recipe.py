from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from ase.build import bulk
import refsite_mlip.config.automatic_reference as automatic_reference_module

from refsite_mlip.config import (
    RecipeReferenceConfig,
    TrainingRecipeConfig,
    TrainingRecipeError,
    resolve_auto_maximum_strain,
)
from refsite_mlip.config.automatic_reference import (
    _PHASE_MAXIMUM_RESIDUAL,
    _automatic_phase,
    _canonical_reference_content,
)
from refsite_mlip.phase import phase_gradient_hessian
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.phase.stabilizer import find_typed_stabilizer


def _rocksalt_self_reference(
    *, repeat: int = 2, nonorthogonal: bool = False
):
    atoms = bulk(
        "NbC", "rocksalt", a=4.4823142441555845, cubic=True
    ).repeat((repeat, repeat, repeat))
    if nonorthogonal:
        cell = atoms.cell.array.copy()
        cell[1, 0] = 0.17320508075688773
        cell[2, 0] = -0.09128709291752768
        cell[2, 1] = 0.1414213562373095
        atoms.set_cell(cell, scale_atoms=True)
    _, numbers, fractional, cell = _canonical_reference_content(atoms)
    species = tuple(sorted(set(int(value) for value in numbers.tolist())))
    type_by_species = {number: index for index, number in enumerate(species)}
    site_types = torch.tensor(
        [type_by_species[int(value)] for value in numbers.tolist()], dtype=torch.long
    )
    stabilizer = find_typed_stabilizer(fractional, site_types)
    return fractional, site_types, cell, stabilizer, len(species)


def _self_phase_snapshot(*, repeat: int = 2, nonorthogonal: bool = False):
    fractional, site_types, cell, stabilizer, num_types = (
        _rocksalt_self_reference(repeat=repeat, nonorthogonal=nonorthogonal)
    )
    specification, certificate = _automatic_phase(
        fractional, site_types, cell, stabilizer, num_types
    )
    return specification, certificate


def _payload_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _count_phase_derivative_calls(monkeypatch):
    original = automatic_reference_module.phase_gradient_hessian
    calls = []

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(
            {
                "cross": args[1].detach().clone(),
                "gradient": result[0].detach().clone(),
                "hessian": result[1].detach().clone(),
            }
        )
        return result

    monkeypatch.setattr(
        automatic_reference_module, "phase_gradient_hessian", counted
    )
    return calls


def test_reference_scalar_list_and_advanced_forms_are_strict():
    scalar = RecipeReferenceConfig.from_dict("pos/POSCAR_222")
    assert scalar.to_dict() == "pos/POSCAR_222"
    assert scalar.sources[0].is_automatic
    assert scalar.sources[0].maximum_strain == "auto"

    listed = RecipeReferenceConfig.from_dict(
        ["pos/POSCAR_222", "pos/POSCAR_333"]
    )
    assert listed.to_dict() == ["pos/POSCAR_222", "pos/POSCAR_333"]

    advanced = RecipeReferenceConfig.from_dict(
        {
            "sources": [
                {
                    "poscar": "pos/POSCAR_222",
                    "template_id": "alpha",
                    "maximum_strain": "auto",
                },
                {
                    "poscar": "pos/POSCAR_333",
                    "template_id": "zeta",
                    "maximum_strain": 0.03,
                },
            ]
        }
    )
    assert advanced.to_dict() == {
        "sources": [
            {"poscar": "pos/POSCAR_222", "template_id": "alpha"},
            {
                "poscar": "pos/POSCAR_333",
                "template_id": "zeta",
                "maximum_strain": 0.03,
            },
        ]
    }


def test_explicit_reference_serialization_is_unchanged():
    payload = {
        "specification": "reference-specification.yaml",
        "poscar": "POSCAR",
        "allow_provisional_phase": True,
    }
    assert RecipeReferenceConfig.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize(
    "value, reason",
    [
        ([], "EMPTY_REFERENCE_SEQUENCE"),
        (True, "INVALID_RECIPE_SECTION"),
        ({"sources": []}, "EMPTY_REFERENCE_SEQUENCE"),
        ({"sources": [{"poscar": "P", "unexpected": 1}]}, "UNKNOWN_RECIPE_KEY"),
        (
            {"specification": "s.yaml", "poscar": "P", "maximum_strain": 0.02},
            "UNKNOWN_RECIPE_KEY",
        ),
        ({"sources": [{"poscar": "P", "maximum_strain": True}]}, "INVALID_MAXIMUM_STRAIN"),
        ({"sources": [{"poscar": "P", "maximum_strain": math.nan}]}, "INVALID_MAXIMUM_STRAIN"),
        ({"sources": [{"poscar": "P", "maximum_strain": math.inf}]}, "INVALID_MAXIMUM_STRAIN"),
    ],
)
def test_invalid_automatic_reference_inputs_are_rejected(value, reason):
    with pytest.raises(TrainingRecipeError) as captured:
        RecipeReferenceConfig.from_dict(value)
    assert captured.value.reason_code == reason


def test_auto_maximum_strain_uses_versioned_integer_tick_policy():
    assert resolve_auto_maximum_strain(0.0) == 0.01
    assert resolve_auto_maximum_strain(0.0175207027) == 0.025
    assert resolve_auto_maximum_strain(0.020000000000000000) == 0.025
    assert resolve_auto_maximum_strain(0.020000000000000004) == 0.025
    assert resolve_auto_maximum_strain(0.045) == 0.05
    assert resolve_auto_maximum_strain(0.0450000001) > 0.05
    for invalid in (True, -0.1, math.nan, math.inf):
        with pytest.raises((TypeError, ValueError)):
            resolve_auto_maximum_strain(invalid)


def test_poscar_only_recipe_defaults_inference_to_sinkhorn():
    payload = {
        "schema_version": "refsite_training_recipe_v1",
        "name": "automatic",
        "model": {},
        "reference": "POSCAR",
        "data": {"train": "train.xyz", "validation": "validation.xyz"},
        "loss": {"energy_weight": 1.0},
        "baseline": "zero",
        "training": {"max_epochs": 1},
        "runtime": {"seed": 1},
    }
    recipe = TrainingRecipeConfig.from_dict(payload)
    assert recipe.ot_solver.to_dict() == {
        "training": "sinkhorn",
        "inference": "sinkhorn",
    }
    with pytest.raises(TrainingRecipeError) as captured:
        TrainingRecipeConfig.from_dict(
            {
                **payload,
                "reference": {
                    "specification": "reference.yaml",
                    "poscar": "POSCAR",
                },
            }
        )
    assert captured.value.reason_code == "MISSING_RECIPE_KEY"


def test_m64_rocksalt_self_reference_recovers_only_failed_legacy_residual(
    monkeypatch,
):
    calls = _count_phase_derivative_calls(monkeypatch)
    specification, certificate = _self_phase_snapshot()
    repeated_specification, repeated_certificate = _self_phase_snapshot()

    assert certificate["certificate_thresholds"]["maximum_residual"] == 1.0e-10
    assert _PHASE_MAXIMUM_RESIDUAL == 1.0e-10
    assert float(torch.linalg.vector_norm(calls[0]["gradient"])) > 1.0e-10
    assert len(calls) == 4
    assert float(torch.linalg.vector_norm(calls[1]["gradient"])) == 0.0
    assert float(torch.linalg.vector_norm(calls[2]["gradient"])) > 1.0e-10
    assert float(torch.linalg.vector_norm(calls[3]["gradient"])) == 0.0
    assert certificate["final_residual"] <= 1.0e-12
    assert specification.to_dict() == repeated_specification.to_dict()
    assert certificate == repeated_certificate


def test_legacy_exact_zero_and_tiny_nonzero_passing_fixtures_do_not_recover(
    monkeypatch,
):
    calls = _count_phase_derivative_calls(monkeypatch)
    exact_specification, exact_certificate = _self_phase_snapshot(repeat=1)
    assert len(calls) == 1
    assert exact_certificate["final_residual"] == 0.0
    assert _payload_sha256(exact_specification.to_dict()) == (
        "ac48036121f006795935dca8072d56537ff63b9b3e0d00b600d53b849b26c016"
    )
    assert _payload_sha256(exact_certificate) == (
        "9565e2bc8db550108e2c105c7a2a3890b846461a458bf7f3b2f5b76ecdd9d50a"
    )

    fractional, site_types, cell, _, num_types = _rocksalt_self_reference()
    translated = fractional + torch.tensor(
        [0.125, -0.25, 1.5], dtype=torch.float64
    )
    translated -= torch.floor(translated)
    stabilizer = find_typed_stabilizer(translated, site_types)
    passing_specification, passing_certificate = _automatic_phase(
        translated, site_types, cell, stabilizer, num_types
    )
    assert len(calls) == 2
    assert 0.0 < passing_certificate["final_residual"] < 1.0e-10
    assert passing_certificate["final_residual"] == 9.522340285319811e-11
    assert _payload_sha256(passing_specification.to_dict()) == (
        "e60a6646e0f775491c485071f9a439840f29dd2c6c116b394c1b9e0f43a63dab"
    )
    assert _payload_sha256(passing_certificate) == (
        "32d88e6ea800458409afd2f1ad67bbb239808aa3e91f88b4ec79933a1428e658"
    )


def test_recovered_exact_cross_is_used_by_the_entire_phase_certificate(
    monkeypatch,
):
    derivative_calls = _count_phase_derivative_calls(monkeypatch)
    original_objective = automatic_reference_module.phase_objective
    objective_calls = []

    def captured_objective(phase, cross, modes, mode_weights):
        objective_calls.append(
            {
                "phase": phase.detach().clone(),
                "cross": cross.detach().clone(),
            }
        )
        return original_objective(phase, cross, modes, mode_weights)

    monkeypatch.setattr(
        automatic_reference_module, "phase_objective", captured_objective
    )
    fractional, site_types, cell, stabilizer, num_types = (
        _rocksalt_self_reference()
    )
    specification, certificate = _automatic_phase(
        fractional, site_types, cell, stabilizer, num_types
    )

    assert len(derivative_calls) == 2
    exact_cross = derivative_calls[1]["cross"]
    assert not torch.equal(derivative_calls[0]["cross"], exact_cross)
    assert objective_calls
    assert all(torch.equal(call["cross"], exact_cross) for call in objective_calls)
    assert torch.equal(objective_calls[0]["phase"], torch.zeros(3, dtype=torch.float64))

    exact_gradient = derivative_calls[1]["gradient"]
    exact_hessian = derivative_calls[1]["hessian"]
    exact_curvature = torch.linalg.eigvalsh(-exact_hessian)
    assert certificate["final_residual"] == float(
        torch.linalg.vector_norm(exact_gradient)
    )
    assert certificate["hessian_minimum_curvature"] == float(
        exact_curvature.min()
    )
    assert certificate["hessian_condition"] == float(
        exact_curvature.max() / exact_curvature.min()
    )

    baseline = float(
        original_objective(
            torch.zeros(3, dtype=torch.float64),
            exact_cross,
            specification.modes,
            specification.mode_weights,
        )
    )
    competitor_scores = [
        float(
            original_objective(
                call["phase"],
                exact_cross,
                specification.modes,
                specification.mode_weights,
            )
        )
        for call in objective_calls[1:]
    ]
    assert certificate["non_equivalent_candidate_objective_gap"] == (
        baseline - max(competitor_scores)
    )


@pytest.mark.parametrize(
    "hessian",
    [
        torch.eye(3, dtype=torch.float64),
        torch.diag(
            torch.tensor([-1.0e-9, -1.0e4, -1.0e4], dtype=torch.float64)
        ),
    ],
)
def test_residual_recovery_cannot_bypass_hessian_certificates(
    monkeypatch, hessian
):
    calls = 0

    def invalid_certificate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return torch.ones(3, dtype=torch.float64), hessian.clone()

    monkeypatch.setattr(
        automatic_reference_module,
        "phase_gradient_hessian",
        invalid_certificate,
    )
    fractional, site_types, cell, stabilizer, num_types = (
        _rocksalt_self_reference()
    )
    with pytest.raises(automatic_reference_module.AutomaticReferenceError) as caught:
        _automatic_phase(fractional, site_types, cell, stabilizer, num_types)
    assert caught.value.reason_code == "PHASE_HESSIAN_CERTIFICATE_FAILED"
    assert calls == 1


def test_residual_recovery_cannot_bypass_candidate_gap_certificate(monkeypatch):
    derivative_calls = _count_phase_derivative_calls(monkeypatch)

    def tied_objective(phase, cross, modes, mode_weights):
        return torch.zeros((), dtype=phase.dtype)

    monkeypatch.setattr(
        automatic_reference_module, "phase_objective", tied_objective
    )
    fractional, site_types, cell, stabilizer, num_types = (
        _rocksalt_self_reference()
    )
    with pytest.raises(automatic_reference_module.AutomaticReferenceError) as caught:
        _automatic_phase(fractional, site_types, cell, stabilizer, num_types)
    assert caught.value.reason_code == "PHASE_GAP_CERTIFICATE_FAILED"
    assert len(derivative_calls) == 2


def test_m64_self_reference_is_stable_to_threads_order_and_translation():
    specification, certificate = _self_phase_snapshot()
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one_thread_specification, one_thread_certificate = _self_phase_snapshot()
    finally:
        torch.set_num_threads(original_threads)
    assert one_thread_specification.to_dict() == specification.to_dict()
    assert one_thread_certificate == certificate

    fractional, site_types, cell, _, num_types = _rocksalt_self_reference()
    order = torch.arange(fractional.shape[0] - 1, -1, -1)
    reordered_fractional = fractional[order].contiguous()
    reordered_types = site_types[order].contiguous()
    reordered_stabilizer = find_typed_stabilizer(
        reordered_fractional, reordered_types
    )
    reordered_specification, reordered_certificate = _automatic_phase(
        reordered_fractional,
        reordered_types,
        cell,
        reordered_stabilizer,
        num_types,
    )
    assert torch.equal(reordered_specification.modes, specification.modes)
    assert reordered_certificate["final_residual"] <= 1.0e-12

    translated_fractional = fractional + torch.tensor(
        [0.125, -0.25, 1.5], dtype=torch.float64
    )
    translated_fractional -= torch.floor(translated_fractional)
    translated_stabilizer = find_typed_stabilizer(
        translated_fractional, site_types
    )
    translated_specification, translated_certificate = _automatic_phase(
        translated_fractional,
        site_types,
        cell,
        translated_stabilizer,
        num_types,
    )
    assert torch.equal(translated_specification.modes, specification.modes)
    assert 0.0 < translated_certificate["final_residual"] < 1.0e-10


def test_self_reference_nonorthogonal_cell_and_distinct_input_negative_case():
    _, nonorthogonal_certificate = _self_phase_snapshot(nonorthogonal=True)
    assert nonorthogonal_certificate["final_residual"] <= 1.0e-12

    fractional, site_types, cell, _, num_types = _rocksalt_self_reference()
    specification, _ = _self_phase_snapshot()
    perturbed = fractional.clone()
    perturbed[0, 0] += 1.0e-5
    identity = torch.eye(num_types, dtype=torch.float64)
    _, _, cross = typed_reciprocal_fields(
        perturbed @ cell,
        torch.zeros(3, dtype=torch.float64),
        cell,
        fractional,
        identity[site_types],
        identity[site_types],
        specification.modes,
        specification.channel_weights,
    )
    gradient, _ = phase_gradient_hessian(
        torch.zeros(3, dtype=torch.float64),
        cross,
        specification.modes,
        specification.mode_weights,
    )
    assert float(torch.linalg.vector_norm(gradient)) > _PHASE_MAXIMUM_RESIDUAL


def test_m64_self_reference_is_stable_across_fresh_processes():
    root = Path(__file__).resolve().parents[2]
    script = """
import json
import torch
from ase.build import bulk
from refsite_mlip.config.automatic_reference import (
    _automatic_phase, _canonical_reference_content,
)
from refsite_mlip.phase.stabilizer import find_typed_stabilizer
atoms = bulk('NbC', 'rocksalt', a=4.4823142441555845, cubic=True).repeat((2, 2, 2))
_, numbers, fractional, cell = _canonical_reference_content(atoms)
species = tuple(sorted(set(int(value) for value in numbers.tolist())))
lookup = {number: index for index, number in enumerate(species)}
site_types = torch.tensor([lookup[int(value)] for value in numbers.tolist()], dtype=torch.long)
stabilizer = find_typed_stabilizer(fractional, site_types)
specification, certificate = _automatic_phase(
    fractional, site_types, cell, stabilizer, len(species)
)
print(json.dumps({
    'specification': specification.to_dict(),
    'certificate': certificate,
}, sort_keys=True, separators=(',', ':')))
"""
    environment = dict(os.environ)
    source = str(root / "src")
    environment["PYTHONPATH"] = source
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    payload = json.loads(outputs[0])
    assert payload["certificate"]["final_residual"] <= 1.0e-12
