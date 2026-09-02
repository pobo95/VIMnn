from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

import refsite_mlip.cli.evaluate as evaluate_module
from refsite_mlip.cli.errors import CLIError, format_cli_error
from refsite_mlip.cli.evaluate import (
    ExtXYZEvaluationConfig,
    _full_prediction,
    _input_semantic_digest,
    _loss_report,
    _physical_metrics,
    _prepare_labeled_samples,
    _write_atomic_json,
    normalize_terms,
    render_evaluation_human,
    render_evaluation_json,
)
from refsite_mlip.cli.main import build_parser, main
from refsite_mlip.data import StructureSample
from refsite_mlip.training import compute_potential_loss
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED


def _config(tmp_path, **overrides):
    values = {
        "bundle_path": str(tmp_path / "model.pt"),
        "input_path": str(tmp_path / "input.xyz"),
    }
    values.update(overrides)
    return ExtXYZEvaluationConfig(**values)


def _geometry(index: int = 0, template_id: str = "zeta", count: int = 2):
    return StructureSample(
        sample_id=f"evaluate:{index:06d}",
        positions=torch.arange(count * 3, dtype=torch.float64).reshape(count, 3)
        / 10.0,
        atomic_numbers=torch.tensor(([6, 41] * count)[:count], dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float64) * 4.0,
        pbc=torch.ones(3, dtype=torch.bool),
        origin=torch.zeros(3, dtype=torch.float64),
        template_id=template_id,
    )


def _oracle_batch():
    force_mask = torch.tensor(
        [[True, True, True], [True, False, True], [False, False, False]]
    )
    stress_mask = torch.zeros((2, 3, 3), dtype=torch.bool)
    stress_mask[0] = True
    return SimpleNamespace(
        energy=torch.tensor([1.0, 4.0], dtype=torch.float64),
        energy_mask=torch.tensor([True, True]),
        atom_ptr=torch.tensor([0, 1, 3], dtype=torch.long),
        atom_batch=torch.tensor([0, 1, 1], dtype=torch.long),
        forces=torch.zeros((3, 3), dtype=torch.float64),
        force_mask=force_mask,
        force_present=torch.tensor([True, True]),
        stress=torch.zeros((2, 3, 3), dtype=torch.float64),
        stress_mask=stress_mask,
        stress_present=torch.tensor([True, False]),
    )


def _oracle_prediction():
    stress = torch.tensor(
        [[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]],
        dtype=torch.float64,
    )
    return SimpleNamespace(
        energy=torch.tensor([3.0, 0.0], dtype=torch.float64),
        forces=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float64,
        ),
        stress=torch.stack((stress, torch.full((3, 3), 99.0, dtype=torch.float64))),
    )


def test_config_parser_terms_scales_weights_and_usage(tmp_path, capsys):
    config = _config(
        tmp_path,
        solver_path="eval-adaptive",
        terms="stress,energy",
        dtype="float32",
        energy_mode="per-atom",
        energy_scale=2,
        force_scale=3,
        stress_scale=4,
        energy_weight=0,
        force_weight=2,
        stress_weight=3,
    )
    assert config.solver_path == EVAL_ADAPTIVE
    assert config.terms == ("energy", "stress")
    assert config.dtype == torch.float32
    assert config.energy_mode == "per_atom"
    assert not config.compute_forces and config.compute_stress
    assert config.operation_name == "evaluation"
    assert config.sample_id_prefix == "evaluate"
    loss = config.loss_config()
    assert loss.energy_weight == 0.0
    assert loss.force_weight == 0.0
    assert loss.stress_weight == 3.0
    assert normalize_terms("stress,forces") == ("forces", "stress")

    parser = build_parser()
    args = parser.parse_args(
        ["evaluate", "--bundle", "m.pt", "--input", "v.xyz"]
    )
    assert args.index == ":"
    assert args.solver == "train-fixed"
    assert args.terms == ("energy", "forces")
    assert args.energy_mode == "per-structure"
    assert args.batch_size == 8
    assert args.output_path is None

    base = ["evaluate", "--bundle", "m.pt", "--input", "v.xyz"]
    for extra in (
        ["--terms", "energy,dipole"],
        ["--terms", "energy,energy"],
        ["--batch-size", "0"],
        ["--energy-scale", "0"],
        ["--force-weight", "-1"],
        ["--stress-scale", "nan"],
        ["--template-id", "zeta", "--template-key", "template"],
    ):
        with pytest.raises(SystemExit) as caught:
            main([*base, *extra])
        assert caught.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "usage:" in captured.err

    with pytest.raises(ValueError, match="device"):
        _config(tmp_path, device="cpu:0")
    with pytest.raises(ValueError, match="overwrite"):
        _config(tmp_path, overwrite=True)


def test_manual_physical_metric_oracle_and_stress_factor_two():
    metrics = _physical_metrics(
        _oracle_prediction(),
        _oracle_batch(),
        ("energy", "forces", "stress"),
    )
    assert metrics["energy"]["total"]["mae"] == pytest.approx(3.0)
    assert metrics["energy"]["total"]["rmse"] == pytest.approx(math.sqrt(10.0))
    assert metrics["energy"]["per_atom"]["mae"] == pytest.approx(2.0)
    assert metrics["energy"]["per_atom"]["rmse"] == pytest.approx(2.0)
    assert metrics["energy"]["valid_structures"] == 2

    force = metrics["forces"]
    assert force["components"]["mae"] == pytest.approx(16.0 / 5.0)
    assert force["components"]["rmse"] == pytest.approx(math.sqrt(66.0 / 5.0))
    assert force["valid_components"] == 5
    assert force["valid_atoms"] == 1
    assert force["vector_error"]["mean"] == pytest.approx(math.sqrt(14.0))
    assert force["vector_error"]["max"] == pytest.approx(math.sqrt(14.0))

    stress = metrics["stress"]
    assert stress["components"]["mae"] == pytest.approx(21.0 / 6.0)
    assert stress["components"]["rmse"] == pytest.approx(math.sqrt(91.0 / 6.0))
    assert stress["frobenius_numerator"] == pytest.approx(168.0)
    assert stress["frobenius_mean"] == pytest.approx(28.0)
    assert stress["valid_independent_components"] == 6


def test_compute_potential_loss_oracle_energy_modes_weights_and_scales(tmp_path):
    from test_potential_losses import _batch

    batch = _batch(
        (1, 2),
        energy=(1.0, 4.0),
        energy_mask=(True, True),
        forces=torch.zeros((3, 3)),
        force_mask=_oracle_batch().force_mask,
        force_present=(True, True),
        stress=torch.zeros((2, 3, 3)),
        stress_mask=_oracle_batch().stress_mask,
        stress_present=(True, False),
    )
    prediction = _oracle_prediction()
    per_structure_config = _config(
        tmp_path,
        terms=("energy", "forces", "stress"),
        energy_scale=2.0,
        force_scale=2.0,
        stress_scale=2.0,
        energy_weight=2.0,
        force_weight=3.0,
        stress_weight=4.0,
    )
    loss = compute_potential_loss(
        prediction, batch, per_structure_config.loss_config()
    )
    assert float(loss.energy.numerator) == pytest.approx(5.0)
    assert float(loss.energy.denominator) == 2.0
    assert float(loss.force.numerator) == pytest.approx(16.5)
    assert float(loss.force.denominator) == 5.0
    assert float(loss.stress.numerator) == pytest.approx(42.0)
    assert float(loss.stress.denominator) == 6.0
    assert float(loss.total) == pytest.approx(42.9)
    reported = _loss_report(loss, per_structure_config)
    assert reported["total_normalized"] == pytest.approx(42.9)
    assert reported["terms"]["stress"]["valid_count"] == 6

    per_atom_config = _config(
        tmp_path,
        terms=("energy",),
        energy_mode="per-atom",
    )
    per_atom = compute_potential_loss(
        SimpleNamespace(energy=prediction.energy),
        batch,
        per_atom_config.loss_config(),
    )
    assert float(per_atom.energy.numerator) == pytest.approx(8.0)
    assert float(per_atom.energy.mean) == pytest.approx(4.0)


def _labeled_atoms(*, zero: bool = True):
    atoms = Atoms(
        numbers=[6, 41],
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=0.0 if zero else 2.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    atoms.arrays["force_mask"] = np.array(
        [[True, False, True], [True, True, True]], dtype=bool
    )
    atoms.info["stress_mask"] = np.array(
        [True, False, True, False, True, False], dtype=bool
    )
    return atoms


def test_label_loading_masks_missing_and_real_zero_are_distinct(tmp_path):
    first = _labeled_atoms(zero=True)
    missing = _labeled_atoms()
    missing.calc = SimpleNamespace(results={})
    missing.arrays.pop("force_mask")
    missing.info.pop("stress_mask")
    geometry = (_geometry(0), _geometry(1))
    config = _config(tmp_path, terms=("energy", "forces", "stress"))
    samples = _prepare_labeled_samples(
        (first, missing), geometry, config, Path(config.input_path)
    )
    assert samples[0].energy is not None and float(samples[0].energy) == 0.0
    assert samples[1].energy is None
    assert torch.equal(
        samples[0].force_mask,
        torch.tensor([[True, False, True], [True, True, True]]),
    )
    assert torch.equal(
        samples[0].stress_mask,
        torch.tensor(
            [[True, False, True], [False, False, False], [True, False, True]]
        ),
    )

    conflict = _labeled_atoms()
    conflict.info["energy"] = 9.0
    with pytest.raises(CLIError) as caught:
        _prepare_labeled_samples(
            (conflict,),
            (_geometry(0),),
            _config(tmp_path, terms=("energy",)),
            Path(config.input_path),
        )
    assert caught.value.reason_code == "CONFLICTING_LABEL"
    assert caught.value.term == "energy"
    assert caught.value.frame_index == 0


def test_all_missing_or_all_masked_requested_term_fails_with_context(tmp_path):
    atoms = _labeled_atoms()
    atoms.calc.results.pop("forces")
    atoms.arrays.pop("force_mask")
    config = _config(tmp_path, terms=("forces",))
    with pytest.raises(CLIError) as caught:
        _prepare_labeled_samples(
            (atoms,), (_geometry(0),), config, Path(config.input_path)
        )
    assert caught.value.reason_code == "NO_VALID_LABELS"
    assert caught.value.term == "forces"
    assert caught.value.sample_id == "evaluate:000000"
    assert caught.value.template_id == "zeta"
    assert caught.value.solver_path == TRAIN_FIXED

    masked = _labeled_atoms()
    masked.arrays["force_mask"][:] = False
    with pytest.raises(CLIError) as caught:
        _prepare_labeled_samples(
            (masked,), (_geometry(0),), config, Path(config.input_path)
        )
    assert caught.value.reason_code == "NO_VALID_LABELS"


def test_prediction_combination_and_derivative_presence(tmp_path):
    structures = tuple(
        SimpleNamespace(
            energy=torch.tensor(float(index), dtype=torch.float64),
            forces=torch.full((index + 1, 3), float(index), dtype=torch.float64),
            stress=torch.eye(3, dtype=torch.float64) * index,
        )
        for index in (1, 2)
    )
    full = _full_prediction(
        structures,
        _config(tmp_path, terms=("energy", "forces", "stress")),
    )
    assert full.energy.tolist() == [1.0, 2.0]
    assert full.forces.shape == (5, 3)
    assert full.stress.shape == (2, 3, 3)
    energy_only = _full_prediction(
        structures, _config(tmp_path, terms=("energy",))
    )
    assert energy_only.forces is None and energy_only.stress is None


def _report():
    return {
        "bundle_sha256": "a" * 64,
        "composition": [],
        "conventions": {
            "energy_unit": "eV",
            "force_unit": "eV/angstrom",
            "stress_sign": "tensile_positive",
            "stress_unit": "eV/angstrom^3",
            "stress_voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        },
        "device": "cpu",
        "dtype": "float64",
        "energy_mode": "per-atom",
        "frame_count": 1,
        "input_semantic_sha256": "b" * 64,
        "labels": {
            "energy": {
                "missing_frames": 0,
                "present_frames": 1,
                "valid_count": 1,
                "valid_count_kind": "structures",
            },
            "forces": {
                "missing_frames": 0,
                "present_frames": 1,
                "valid_count": 6,
                "valid_count_kind": "cartesian_components",
            },
            "stress": {
                "missing_frames": 0,
                "present_frames": 1,
                "valid_count": 6,
                "valid_count_kind": "independent_voigt_components",
            },
        },
        "loss": {
            "terms": {
                term: {
                    "denominator": 1.0,
                    "mean": 0.25,
                    "numerator": 0.25,
                    "valid_count": 1,
                    "weight": 1.0,
                }
                for term in ("energy", "forces", "stress")
            },
            "total_normalized": 2.5,
        },
        "metrics": {
            "energy": {
                "per_atom": {"mae": 0.5, "rmse": 0.75, "unit": "eV/atom"},
                "total": {"mae": 1.0, "rmse": 1.5, "unit": "eV"},
                "valid_structures": 1,
            },
            "forces": {
                "components": {
                    "mae": 0.2,
                    "rmse": 0.3,
                    "unit": "eV/angstrom",
                },
                "valid_atoms": 2,
                "valid_components": 6,
                "vector_error": {
                    "max": 0.5,
                    "mean": 0.4,
                    "unit": "eV/angstrom",
                },
            },
            "stress": {
                "components": {
                    "mae": 0.01,
                    "rmse": 0.02,
                    "unit": "eV/angstrom^3",
                },
                "frobenius_mean": 0.03,
                "frobenius_numerator": 0.18,
                "valid_independent_components": 6,
            },
        },
        "requested_terms": ["energy", "forces", "stress"],
        "scales": {"energy": 1.0, "forces": 1.0, "stress": 1.0},
        "solver": "train-fixed",
        "template_frame_counts": {"alpha": 1},
        "weights": {"energy": 1.0, "forces": 1.0, "stress": 1.0},
    }


def test_deterministic_json_human_and_semantic_digest():
    report = _report()
    first = render_evaluation_json(report)
    second = render_evaluation_json(dict(reversed(tuple(report.items()))))
    assert first == second
    assert json.loads(first) == report
    human = render_evaluation_human(report)
    assert "Frames: 1" in human
    assert "Templates: alpha=1" in human
    assert "Energy:" in human and "Forces:" in human
    assert "tensile-positive, xx yy zz yz xz xy" in human
    assert "Normalized loss stress:" in human
    assert "Total normalized loss: 2.5" in human

    sample = replace_sample = _geometry(0)
    digest = _input_semantic_digest((sample,), ("energy",))
    assert digest == _input_semantic_digest((replace_sample,), ("energy",))
    changed = StructureSample(
        sample_id=sample.sample_id,
        positions=sample.positions + 0.01,
        atomic_numbers=sample.atomic_numbers,
        cell=sample.cell,
        pbc=sample.pbc,
        origin=sample.origin,
        template_id=sample.template_id,
    )
    assert digest != _input_semantic_digest((changed,), ("energy",))


def test_atomic_json_output_failure_preserves_target_and_cleans_temp(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.xyz"
    source.write_text("input")
    target = tmp_path / "report.json"
    target.write_bytes(b"old report")
    config = _config(
        tmp_path,
        output_path=target,
        overwrite=True,
    )

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise OSError("injected")

    monkeypatch.setattr(evaluate_module.os, "replace", fail_replace)
    with pytest.raises(CLIError) as caught:
        _write_atomic_json(
            target,
            render_evaluation_json(_report()),
            source=source.resolve(),
            config=config,
        )
    assert caught.value.reason_code == "OUTPUT_COMMIT_FAILED"
    assert target.read_bytes() == b"old report"
    assert source.read_text() == "input"
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_term_error_context_is_concise():
    error = CLIError(
        "NO_VALID_LABELS",
        "requested evaluation term has no valid labels",
        stage="evaluation.label_preflight",
        path="validation.xyz",
        frame_index=2,
        sample_id="evaluate:000002",
        template_id="zeta",
        term="stress",
        solver_path=EVAL_ADAPTIVE,
        prediction_stage="label_preflight",
        predictor_reason_code="NO_VALID_LABELS",
    )
    rendered = format_cli_error(error)
    for value in (
        "frame_index=2",
        "sample_id='evaluate:000002'",
        "template_id='zeta'",
        "term='stress'",
        "solver_path='eval_adaptive'",
        "predictor_reason_code='NO_VALID_LABELS'",
    ):
        assert value in rendered
