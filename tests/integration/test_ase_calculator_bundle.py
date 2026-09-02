from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes

import refsite_mlip.interfaces.ase_calculator as ase_module
from refsite_mlip.data import StructureSample
from refsite_mlip.inference import PredictorError, ReferenceSitePredictor
from refsite_mlip.interfaces import (
    ASECalculatorConfig,
    ReferenceSiteASECalculator,
    ReferenceSiteASECalculatorError,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from test_bundle_predictor_runtime import _save_case


def _atoms(sample: StructureSample) -> Atoms:
    return Atoms(
        numbers=sample.atomic_numbers.detach().cpu().numpy().copy(),
        positions=sample.positions.detach().cpu().numpy().copy(),
        cell=sample.cell.detach().cpu().numpy().copy(),
        pbc=sample.pbc.detach().cpu().numpy().copy(),
    )


def _sample_from_atoms(atoms: Atoms, template_id: str) -> StructureSample:
    return StructureSample(
        sample_id="ase:structure",
        positions=torch.tensor(atoms.get_positions(), dtype=torch.float64),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        cell=torch.tensor(atoms.cell.array, dtype=torch.float64),
        pbc=torch.tensor(atoms.get_pbc(), dtype=torch.bool),
        origin=torch.zeros(3, dtype=torch.float64),
        template_id=template_id,
    )


def _assert_numpy_results(calculator: ReferenceSiteASECalculator) -> None:
    assert set(calculator.results) <= {
        "energy",
        "free_energy",
        "forces",
        "stress",
    }
    for value in calculator.results.values():
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value))
        else:
            assert isinstance(value, float) and np.isfinite(value)


def test_cpu_float64_predictor_parity_pristine_vacancy_and_stress_convention(
    typed_crystal, tmp_path
):
    *_, samples, _, _, _, _, path = _save_case(typed_crystal, tmp_path)
    maximum = 0.0
    for solver_path in (TRAIN_FIXED, EVAL_ADAPTIVE):
        for index in (0, 1):
            original = samples[index]
            atoms = _atoms(original)
            calculator = ReferenceSiteASECalculator(
                path,
                template_id=original.template_id,
                dtype=torch.float64,
                solver_path=solver_path,
            )
            direct = calculator.predictor.predict_sample(
                _sample_from_atoms(atoms, original.template_id),
                solver_path=solver_path,
                compute_forces=True,
                compute_stress=True,
                return_aux=True,
            )
            positions = atoms.positions.copy()
            numbers = atoms.numbers.copy()
            cell = atoms.cell.array.copy()
            pbc = atoms.pbc.copy()
            atoms.calc = calculator

            with torch.no_grad():
                energy = atoms.get_potential_energy()
                free_energy = atoms.get_potential_energy(force_consistent=True)
                forces = atoms.get_forces()
                stress_voigt = atoms.get_stress()
                stress_tensor = atoms.get_stress(voigt=False)

            maximum = max(
                maximum,
                abs(energy - float(direct.energy)),
                float(np.max(np.abs(forces - direct.forces.cpu().numpy()))),
                float(
                    np.max(
                        np.abs(stress_voigt - direct.stress_voigt.cpu().numpy())
                    )
                ),
                float(
                    np.max(np.abs(stress_tensor - direct.stress.cpu().numpy()))
                ),
            )
            assert energy == free_energy
            np.testing.assert_allclose(
                forces, direct.forces.cpu().numpy(), atol=4e-13, rtol=4e-13
            )
            np.testing.assert_allclose(
                stress_voigt,
                direct.stress_voigt.cpu().numpy(),
                atol=4e-13,
                rtol=4e-13,
            )
            np.testing.assert_allclose(
                stress_tensor,
                direct.stress.cpu().numpy(),
                atol=4e-13,
                rtol=4e-13,
            )
            assert stress_voigt[3] == pytest.approx(stress_tensor[1, 2])
            assert stress_voigt[4] == pytest.approx(stress_tensor[0, 2])
            assert stress_voigt[5] == pytest.approx(stress_tensor[0, 1])
            assert np.array_equal(atoms.positions, positions)
            assert np.array_equal(atoms.numbers, numbers)
            assert np.array_equal(atoms.cell.array, cell)
            assert np.array_equal(atoms.pbc, pbc)
            _assert_numpy_results(calculator)
    assert maximum < 4e-13


def test_property_dispatch_ase_cache_and_combined_request(
    typed_crystal, tmp_path, monkeypatch
):
    *_, samples, _, _, _, _, path = _save_case(typed_crystal, tmp_path)
    atoms = _atoms(samples[0])
    calculator = ReferenceSiteASECalculator(path, template_id="zeta")
    calls = []
    original = calculator.predictor.predict_sample

    def counted(*args, **kwargs):
        calls.append((kwargs["compute_forces"], kwargs["compute_stress"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(calculator.predictor, "predict_sample", counted)
    atoms.calc = calculator
    atoms.get_potential_energy()
    atoms.get_potential_energy()
    assert calls == [(False, False)]
    atoms.get_forces()
    atoms.get_stress()
    assert calls == [(False, False), (True, False), (False, True)]

    calculator.reset()
    calculator.calculate(
        atoms, properties=("forces", "stress"), system_changes=all_changes
    )
    assert calls[-1] == (True, True)
    assert set(calculator.results) == {
        "energy",
        "free_energy",
        "forces",
        "stress",
    }


def test_calculator_ownership_readonly_diagnostics_and_failure_cache(
    typed_crystal, tmp_path
):
    *_, samples, _, _, _, bundle, path = _save_case(typed_crystal, tmp_path)
    atoms = _atoms(samples[0])
    calculator = ReferenceSiteASECalculator(path, template_id="zeta")
    model = calculator.predictor.model
    parameter_ids = tuple(id(value) for value in model.parameters())
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    bundle_state = {key: value.clone() for key, value in bundle.model_state.items()}
    cpu_rng = torch.get_rng_state().clone()
    atoms_state = (
        atoms.positions.copy(),
        atoms.numbers.copy(),
        atoms.cell.array.copy(),
        atoms.pbc.copy(),
    )

    calculator.calculate(
        atoms, properties=("energy", "forces", "stress"), system_changes=all_changes
    )
    diagnostics = calculator.last_diagnostics
    assert diagnostics is not None
    with pytest.raises(TypeError):
        diagnostics["solver_path"] = "changed"
    _assert_numpy_results(calculator)
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    assert tuple(model.state_dict()) == tuple(state)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
    assert all(torch.equal(bundle.model_state[key], value) for key, value in bundle_state.items())
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    for actual, expected in zip(
        (atoms.positions, atoms.numbers, atoms.cell.array, atoms.pbc), atoms_state
    ):
        assert np.array_equal(actual, expected)

    bad = atoms.copy()
    bad.pbc = (True, False, True)
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        calculator.calculate(bad, properties=("energy",), system_changes=("pbc",))
    assert caught.value.reason_code == "UNSUPPORTED_PBC"
    assert calculator.last_diagnostics is diagnostics
    assert calculator.results == {}

    singular = atoms.copy()
    singular.cell[2] = singular.cell[1]
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        calculator.calculate(singular, properties=("energy",), system_changes=("cell",))
    assert caught.value.reason_code == "SINGULAR_CELL"
    assert calculator.last_diagnostics is diagnostics

    calculator.reset()
    assert calculator.results == {}
    assert calculator.last_diagnostics is None
    assert calculator.candidate_neighbor_state is None


def test_candidate_state_reuse_rebuild_disable_and_transaction(
    typed_crystal, tmp_path
):
    *_, samples, _, _, _, _, path = _save_case(
        typed_crystal, tmp_path, edge_backend=True
    )
    atoms = _atoms(samples[0])
    calculator = ReferenceSiteASECalculator(
        path,
        template_id="zeta",
        config=ASECalculatorConfig(reuse_candidate_state=True),
    )
    assert calculator.candidate_state_enabled
    atoms.calc = calculator
    initial_energy = atoms.get_potential_energy()
    initial_state = calculator.candidate_neighbor_state
    assert initial_state is not None
    initial_fingerprint = initial_state.integrity_fingerprint
    assert (
        calculator.last_diagnostics["candidate_state"]["decision"]["reason_code"]
        == "INITIAL_BUILD"
    )

    atoms.numbers[0] = 41
    atoms.get_potential_energy()
    assert (
        calculator.last_diagnostics["candidate_state"]["adapter_input_reason"]
        == "ATOM_ORDER_CHANGED"
    )
    assert (
        calculator.last_diagnostics["candidate_state"]["decision"]["reason_code"]
        == "INITIAL_BUILD"
    )

    atoms.get_forces()
    assert (
        calculator.last_diagnostics["candidate_state"]["decision"]["reason_code"]
        == "REUSED"
    )
    initial_state.validate_integrity()
    assert initial_state.integrity_fingerprint == initial_fingerprint
    assert calculator.candidate_neighbor_state.reuse_count == initial_state.reuse_count + 1

    atoms.positions[0, 0] += 0.01
    moved_energy = atoms.get_potential_energy()
    assert np.isfinite(moved_energy)
    assert (
        calculator.last_diagnostics["candidate_state"]["decision"]["reason_code"]
        == "REUSED"
    )

    atoms.positions[0, 0] += 0.25
    atoms.get_potential_energy()
    assert (
        calculator.last_diagnostics["candidate_state"]["decision"]["reason_code"]
        == "SKIN_EXHAUSTED"
    )

    atoms.cell[0, 0] *= 1.0001
    atoms.get_potential_energy()
    assert (
        calculator.last_diagnostics["candidate_state"]["adapter_input_reason"]
        == "CELL_CHANGED"
    )
    assert (
        calculator.last_diagnostics["candidate_state"]["decision"]["reason_code"]
        == "INITIAL_BUILD"
    )

    stateless = ReferenceSiteASECalculator(
        path,
        template_id="zeta",
        config=ASECalculatorConfig(reuse_candidate_state=False),
    )
    fresh_atoms = atoms.copy()
    fresh_atoms.calc = stateless
    assert fresh_atoms.get_potential_energy() == pytest.approx(
        atoms.get_potential_energy(), abs=4e-13, rel=4e-13
    )
    assert stateless.candidate_neighbor_state is None

    state_before_failure = calculator.candidate_neighbor_state
    diagnostics_before_failure = calculator.last_diagnostics
    atoms.pbc = (True, False, True)
    with pytest.raises(ReferenceSiteASECalculatorError):
        atoms.get_potential_energy()
    assert (
        calculator.candidate_neighbor_state.integrity_fingerprint
        == state_before_failure.integrity_fingerprint
    )
    assert calculator.last_diagnostics is diagnostics_before_failure
    assert calculator.results == {}
    atoms.pbc = True
    assert np.isfinite(atoms.get_potential_energy())
    assert initial_energy != moved_energy
    calculator.reset()
    assert calculator.results == {}
    assert calculator.candidate_neighbor_state is None
    assert calculator.last_diagnostics is None


def test_template_errors_policy_errors_inference_mode_and_no_builder(
    typed_crystal, tmp_path, monkeypatch
):
    *_, samples, _, _, _, _, path = _save_case(typed_crystal, tmp_path)
    default = ReferenceSiteASECalculator(path)
    assert default.template_id == "zeta"
    alpha = ReferenceSiteASECalculator(path, template_id="alpha")
    alpha_atoms = _atoms(samples[1])
    alpha_atoms.calc = alpha
    assert np.isfinite(alpha_atoms.get_potential_energy())

    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        ReferenceSiteASECalculator(path, template_id="missing")
    assert caught.value.reason_code == "UNKNOWN_TEMPLATE"

    unsupported = _atoms(samples[0])
    unsupported.numbers[0] = 1
    unsupported.calc = default
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        unsupported.get_potential_energy()
    assert caught.value.reason_code == "UNSUPPORTED_SPECIES"
    assert caught.value.atom_count == len(unsupported)
    assert 1 in caught.value.species
    assert caught.value.predictor_stage == "structure_domain_preflight"

    too_many = _atoms(samples[2])
    too_many += Atoms(
        numbers=[int(too_many.numbers[0])],
        positions=[too_many.positions[0] + 0.1],
    )
    too_many.calc = default
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        too_many.get_potential_energy()
    assert caught.value.reason_code == "INVALID_N_GT_M"

    derivative_atoms = _atoms(samples[0])
    derivative_atoms.calc = default
    with torch.inference_mode(), pytest.raises(ReferenceSiteASECalculatorError) as caught:
        derivative_atoms.get_forces()
    assert caught.value.reason_code == "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED"

    no_policy_directory = tmp_path / "no-policy"
    no_policy_directory.mkdir()
    *_, no_policy_path = _save_case(
        typed_crystal, no_policy_directory, policies=False
    )
    adaptive = ReferenceSiteASECalculator(
        no_policy_path, template_id="zeta", solver_path=EVAL_ADAPTIVE
    )
    adaptive_atoms = _atoms(samples[0])
    adaptive_atoms.calc = adaptive
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        adaptive_atoms.get_potential_energy()
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"

    def forbidden(*args, **kwargs):
        raise AssertionError("builder must not run while loading Calculator")

    import refsite_mlip.data.reference_builder as builder_module
    import refsite_mlip.graph.topology as graph_module
    import refsite_mlip.phase.stabilizer as stabilizer_module

    monkeypatch.setattr(builder_module, "build_reference_template_from_poscar", forbidden)
    monkeypatch.setattr(builder_module, "build_reference_template_from_atoms", forbidden)
    monkeypatch.setattr(builder_module, "canonicalize_reference_atoms", forbidden)
    monkeypatch.setattr(graph_module, "build_reference_graph_topology", forbidden)
    monkeypatch.setattr(stabilizer_module, "find_typed_stabilizer", forbidden)
    builder_free = ReferenceSiteASECalculator(path, template_id="zeta")
    assert builder_free.predictor.model.training is False


def test_candidate_state_fingerprint_mismatch_is_contextual_and_transactional(
    typed_crystal, tmp_path
):
    *_, samples, _, _, _, _, path = _save_case(
        typed_crystal, tmp_path, edge_backend=True
    )
    atoms = _atoms(samples[0])
    calculator = ReferenceSiteASECalculator(path, template_id="zeta")
    atoms.calc = calculator
    atoms.get_potential_energy()
    valid_state = calculator.candidate_neighbor_state
    diagnostics = calculator.last_diagnostics
    corrupted = replace(
        valid_state,
        template_fingerprint="0" * 64,
        integrity_fingerprint=None,
    )
    calculator._candidate_neighbor_state = corrupted
    atoms.positions[0, 1] += 0.001
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        atoms.get_potential_energy()
    assert caught.value.reason_code == "TEMPLATE_MISMATCH"
    assert caught.value.predictor_reason_code == "TEMPLATE_MISMATCH"
    assert caught.value.template_id == "zeta"
    assert calculator.last_diagnostics is diagnostics
    assert calculator.results == {}


def test_missing_default_nonfinite_fallback_and_candidate_mismatch_context(
    typed_crystal, tmp_path, monkeypatch
):
    *_, samples, _, _, _, _, path = _save_case(typed_crystal, tmp_path)
    predictor = ase_module.load_reference_site_predictor(path)
    missing_runtime = replace(predictor.runtime, default_template_id="")
    missing_predictor = ReferenceSitePredictor(missing_runtime)
    monkeypatch.setattr(
        ase_module, "load_reference_site_predictor", lambda *args, **kwargs: missing_predictor
    )
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        ReferenceSiteASECalculator(path)
    assert caught.value.reason_code == "AMBIGUOUS_TEMPLATE"

    monkeypatch.undo()
    calculator = ReferenceSiteASECalculator(path, template_id="zeta")
    atoms = _atoms(samples[0])
    original = calculator.predictor.predict_sample

    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, energy=torch.full_like(result.energy, float("nan")))

    monkeypatch.setattr(calculator.predictor, "predict_sample", nonfinite)
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        calculator.calculate(atoms, properties=("energy",), system_changes=all_changes)
    assert caught.value.reason_code == "NONFINITE_OUTPUT"
    assert calculator.results == {}

    def fallback(*args, **kwargs):
        raise PredictorError(
            "DERIVATIVE_FALLBACK_UNSUPPORTED",
            "synthetic fallback",
            sample_id="ase:structure",
            template_id="zeta",
            solver_path=EVAL_ADAPTIVE,
            stage="transport_derivative",
        )

    monkeypatch.setattr(calculator.predictor, "predict_sample", fallback)
    with pytest.raises(ReferenceSiteASECalculatorError) as caught:
        calculator.calculate(atoms, properties=("forces",), system_changes=all_changes)
    assert caught.value.reason_code == "DERIVATIVE_FALLBACK_UNSUPPORTED"
    assert caught.value.predictor_stage == "transport_derivative"
    assert caught.value.requested_properties == ("forces",)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float32_float64_fixed_adaptive_cpu_ase_results(
    typed_crystal, tmp_path
):
    *_, samples, _, _, _, _, path = _save_case(
        typed_crystal, tmp_path, edge_backend=True
    )
    for dtype in (torch.float32, torch.float64):
        for solver_path in (TRAIN_FIXED, EVAL_ADAPTIVE):
            calculator = ReferenceSiteASECalculator(
                path,
                template_id="zeta",
                device="cuda",
                dtype=dtype,
                solver_path=solver_path,
            )
            atoms = _atoms(samples[0])
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            calculator.calculate(
                atoms,
                properties=("energy", "forces", "stress"),
                system_changes=all_changes,
            )
            assert calculator.predictor.device.type == "cuda"
            assert calculator.predictor.dtype == dtype
            assert calculator.candidate_state_enabled
            expected = np.float32 if dtype == torch.float32 else np.float64
            assert calculator.results["forces"].dtype == expected
            assert calculator.results["stress"].dtype == expected
            _assert_numpy_results(calculator)
            assert calculator.last_diagnostics is not None
            assert torch.equal(torch.get_rng_state(), cpu_rng)
            assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
