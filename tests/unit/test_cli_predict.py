from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

import refsite_mlip.cli.predict as predict_module
from refsite_mlip.cli.errors import CLIError, format_cli_error
from refsite_mlip.cli.main import build_parser, main
from refsite_mlip.cli.predict import (
    ExtXYZPredictionConfig,
    _predict_batches,
    _prediction_frames,
    _summary,
    _validate_predictions,
    _write_atomic_extxyz,
    normalize_properties,
    render_prediction_human,
    render_prediction_json,
)
from refsite_mlip.data import StructureSample
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED


def _config(tmp_path, **overrides):
    values = {
        "bundle_path": str(tmp_path / "model.pt"),
        "input_path": str(tmp_path / "input.xyz"),
        "output_path": str(tmp_path / "output.xyz"),
    }
    values.update(overrides)
    return ExtXYZPredictionConfig(**values)


def _sample(index: int, template_id: str = "zeta") -> StructureSample:
    return StructureSample(
        sample_id=f"predict:{index:06d}",
        positions=torch.tensor(
            [[0.1 + index, 0.2, 0.3], [1.1, 1.2, 1.3]],
            dtype=torch.float64,
        ),
        atomic_numbers=torch.tensor([6, 41], dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float64) * 4.0,
        pbc=torch.ones(3, dtype=torch.bool),
        origin=torch.zeros(3, dtype=torch.float64),
        template_id=template_id,
    )


def _prediction(sample: StructureSample, *, forces: bool, stress: bool):
    index = int(sample.sample_id.rsplit(":", 1)[1])
    energy = torch.tensor(-2.0 + 0.5 * index, dtype=torch.float64)
    force = (
        torch.tensor(
            [[0.1 + index, -0.2, 0.3], [-0.4, 0.5, -0.6]],
            dtype=torch.float64,
        )
        if forces
        else None
    )
    stress_voigt = (
        torch.tensor([1.0, 2.0, 3.0, 0.4, -0.5, 0.6], dtype=torch.float64)
        if stress
        else None
    )
    stress_tensor = (
        torch.tensor(
            [[1.0, 0.6, -0.5], [0.6, 2.0, 0.4], [-0.5, 0.4, 3.0]],
            dtype=torch.float64,
        )
        if stress
        else None
    )
    return SimpleNamespace(
        energy=energy,
        forces=force,
        stress=stress_tensor,
        stress_voigt=stress_voigt,
        sample_id=sample.sample_id,
        template_id=sample.template_id,
    )


def _atoms(index: int = 0) -> Atoms:
    atoms = Atoms(
        numbers=[6, 41],
        positions=[[0.1 + index, 0.2, 0.3], [1.1, 1.2, 1.3]],
        cell=np.eye(3) * 4.0,
        pbc=True,
        info={"keep": "metadata", "energy": 999.0},
    )
    atoms.arrays["marker"] = np.array([10, 20])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=999.0,
        free_energy=998.0,
        forces=np.ones((2, 3)),
        stress=np.arange(6, dtype=float),
    )
    return atoms


def test_config_property_solver_and_dtype_normalization(tmp_path):
    config = _config(tmp_path)
    assert config.solver_path == TRAIN_FIXED
    assert config.properties == ("energy", "forces")
    assert config.compute_forces and not config.compute_stress
    assert config.dtype == torch.float64 and config.dtype_name == "float64"

    adaptive = _config(
        tmp_path,
        solver_path="eval-adaptive",
        properties=("stress",),
        dtype="float32",
        batch_size=3,
    )
    assert adaptive.solver_path == EVAL_ADAPTIVE
    assert adaptive.properties == ("energy", "stress")
    assert not adaptive.compute_forces and adaptive.compute_stress
    assert normalize_properties("stress,forces") == (
        "energy",
        "forces",
        "stress",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(tmp_path, template_id="zeta", template_key="template")
    with pytest.raises(ValueError, match="duplicates"):
        _config(tmp_path, properties=("energy", "energy"))
    with pytest.raises(ValueError, match="unknown"):
        _config(tmp_path, properties=("dipole",))
    with pytest.raises(ValueError, match="positive"):
        _config(tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="device"):
        _config(tmp_path, device="mps")
    with pytest.raises(ValueError, match="dtype"):
        _config(tmp_path, dtype=torch.float16)


def test_predict_parser_defaults_choices_and_usage_errors(capsys):
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--bundle",
            "model.pt",
            "--input",
            "in.xyz",
            "--output",
            "out.xyz",
        ]
    )
    assert args.index == ":"
    assert args.solver == "train-fixed"
    assert args.properties == ("energy", "forces")
    assert args.device == "cpu"
    assert args.dtype == "float64"
    assert args.batch_size == 8

    invalid = (
        ["--template-id", "zeta", "--template-key", "template"],
        ["--properties", "energy,dipole"],
        ["--batch-size", "0"],
        ["--device", "cuda:x"],
        ["--dtype", "float16"],
    )
    base = [
        "predict",
        "--bundle",
        "model.pt",
        "--input",
        "in.xyz",
        "--output",
        "out.xyz",
    ]
    for extra in invalid:
        with pytest.raises(SystemExit) as caught:
            main([*base, *extra])
        assert caught.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "usage:" in captured.err


@pytest.mark.parametrize(
    ("properties", "expected_forces", "expected_stress"),
    [
        (("energy",), False, False),
        (("energy", "forces"), True, False),
        (("energy", "stress"), False, True),
        (("energy", "forces", "stress"), True, True),
    ],
)
def test_prediction_batches_control_derivatives_and_use_fresh_state(
    tmp_path, properties, expected_forces, expected_stress
):
    samples = tuple(_sample(index) for index in range(5))
    calls = []

    class FakePredictor:
        def predict_samples(self, chunk, **kwargs):
            calls.append((tuple(value.sample_id for value in chunk), dict(kwargs)))
            structures = tuple(
                _prediction(
                    sample,
                    forces=kwargs["compute_forces"],
                    stress=kwargs["compute_stress"],
                )
                for sample in chunk
            )
            return SimpleNamespace(
                sample_ids=tuple(sample.sample_id for sample in chunk),
                structures=structures,
            )

    config = _config(tmp_path, properties=properties, batch_size=2)
    predictions = _predict_batches(
        samples,
        FakePredictor(),
        config,
        Path(config.input_path),
    )
    assert len(predictions) == 5
    assert [len(call[0]) for call in calls] == [2, 2, 1]
    for _, options in calls:
        assert options["compute_forces"] is expected_forces
        assert options["compute_stress"] is expected_stress
        assert options["candidate_neighbor_states"] is None
        assert not options["return_candidate_neighbor_states"]
        assert not options["return_aux"]


def test_output_frames_replace_labels_preserve_geometry_metadata_and_inputs(tmp_path):
    atoms = _atoms()
    original = (
        atoms.positions.copy(),
        atoms.numbers.copy(),
        atoms.cell.array.copy(),
        dict(atoms.info),
        atoms.calc,
        atoms.calc.results["forces"].copy(),
    )
    sample = _sample(0)
    prediction = _prediction(sample, forces=True, stress=True)
    config = _config(tmp_path, properties=("energy", "forces", "stress"))
    output = _prediction_frames(
        (atoms,),
        (prediction,),
        config=config,
        bundle_fingerprint="f" * 64,
    )[0]

    np.testing.assert_array_equal(output.positions, original[0])
    np.testing.assert_array_equal(output.numbers, original[1])
    np.testing.assert_array_equal(output.cell.array, original[2])
    np.testing.assert_array_equal(output.arrays["marker"], [10, 20])
    assert output.info["keep"] == "metadata"
    assert "energy" not in output.info
    assert output.info["refsite_template_id"] == "zeta"
    assert output.info["refsite_solver_path"] == TRAIN_FIXED
    assert output.info["refsite_bundle_sha256"] == "f" * 64
    assert output.calc.results["energy"] == float(prediction.energy)
    assert output.calc.results["free_energy"] == float(prediction.energy)
    np.testing.assert_array_equal(
        output.calc.results["forces"], prediction.forces.numpy()
    )
    np.testing.assert_array_equal(
        output.calc.results["stress"], prediction.stress_voigt.numpy()
    )

    np.testing.assert_array_equal(atoms.positions, original[0])
    np.testing.assert_array_equal(atoms.numbers, original[1])
    np.testing.assert_array_equal(atoms.cell.array, original[2])
    assert atoms.info == original[3]
    assert atoms.calc is original[4]
    np.testing.assert_array_equal(atoms.calc.results["forces"], original[5])


def test_prediction_validation_nonfinite_missing_and_unrequested_derivatives(tmp_path):
    atoms = _atoms()
    sample = _sample(0)
    energy_only = _config(tmp_path, properties=("energy",))
    finite = _prediction(sample, forces=False, stress=False)
    _validate_predictions((atoms,), (sample,), (finite,), energy_only, Path("in.xyz"))

    with pytest.raises(CLIError) as caught:
        nonfinite = SimpleNamespace(**vars(finite))
        nonfinite.energy = torch.tensor(float("nan"))
        _validate_predictions(
            (atoms,),
            (sample,),
            (nonfinite,),
            energy_only,
            Path("in.xyz"),
        )
    assert caught.value.reason_code == "NONFINITE_PREDICTION"
    assert caught.value.frame_index == 0

    with pytest.raises(CLIError) as caught:
        _validate_predictions(
            (atoms,),
            (sample,),
            (_prediction(sample, forces=True, stress=False),),
            energy_only,
            Path("in.xyz"),
        )
    assert caught.value.reason_code == "UNREQUESTED_DERIVATIVE_OUTPUT"


def test_summary_json_human_are_deterministic_and_complete(tmp_path):
    samples = (_sample(0, "zeta"), _sample(1, "alpha"))
    predictions = tuple(
        _prediction(sample, forces=True, stress=True) for sample in samples
    )
    config = _config(
        tmp_path,
        properties=("energy", "forces", "stress"),
        solver_path=EVAL_ADAPTIVE,
    )
    report = _summary(predictions, config, bundle_fingerprint="a" * 64)
    first = render_prediction_json(report)
    second = render_prediction_json(dict(reversed(tuple(report.items()))))
    assert first == second
    restored = json.loads(first)
    assert restored["frame_count"] == 2
    assert restored["template_frame_counts"] == {"alpha": 1, "zeta": 1}
    assert restored["energy"] == {"min": -2.0, "mean": -1.75, "max": -1.5}
    assert restored["forces"]["component_rms"] > 0.0
    assert restored["forces"]["max_force_norm"] > 0.0
    assert restored["stress"]["component_min"] == -0.5
    assert restored["stress"]["component_max"] == 3.0
    assert restored["stress"]["voigt_order"] == [
        "xx",
        "yy",
        "zz",
        "yz",
        "xz",
        "xy",
    ]
    human = render_prediction_human(report)
    assert "Templates: alpha=1, zeta=1" in human
    assert "tensile-positive, xx yy zz yz xz xy" in human


def test_atomic_replace_failure_preserves_target_and_cleans_temporary(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.xyz"
    source.write_text("input remains")
    target = tmp_path / "output.xyz"
    target.write_bytes(b"existing output")
    config = _config(tmp_path, overwrite=True)
    frame = _atoms()
    frame.info.pop("energy")
    frame.calc = None

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise OSError("injected commit failure")

    monkeypatch.setattr(predict_module.os, "replace", fail_replace)
    with pytest.raises(CLIError) as caught:
        _write_atomic_extxyz(
            target,
            (frame,),
            source=source.resolve(),
            config=config,
        )
    assert caught.value.reason_code == "OUTPUT_COMMIT_FAILED"
    assert target.read_bytes() == b"existing output"
    assert source.read_text() == "input remains"
    assert not list(tmp_path.glob(".output.xyz.*.tmp"))


def test_no_overwrite_commit_race_preserves_competing_output(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.xyz"
    source.write_text("input remains")
    target = tmp_path / "output.xyz"
    config = _config(tmp_path, overwrite=False)
    frame = _atoms()
    frame.info.pop("energy")
    frame.calc = None

    def competing_link(temporary, destination):
        del temporary
        Path(destination).write_bytes(b"competing output")
        raise FileExistsError("injected commit race")

    monkeypatch.setattr(predict_module.os, "link", competing_link)
    with pytest.raises(CLIError) as caught:
        _write_atomic_extxyz(
            target,
            (frame,),
            source=source.resolve(),
            config=config,
        )
    assert caught.value.reason_code == "OUTPUT_EXISTS"
    assert target.read_bytes() == b"competing output"
    assert source.read_text() == "input remains"
    assert not list(tmp_path.glob(".output.xyz.*.tmp"))


def test_predictor_error_context_is_rendered_concisely(tmp_path):
    error = CLIError(
        "UNSUPPORTED_SPECIES",
        "Predictor batch execution failed",
        stage="prediction.structure_domain_preflight",
        path=tmp_path / "input.xyz",
        frame_index=3,
        sample_id="predict:000003",
        template_id="zeta",
        solver_path=EVAL_ADAPTIVE,
        prediction_stage="structure_domain_preflight",
        predictor_reason_code="UNSUPPORTED_SPECIES",
    )
    formatted = format_cli_error(error)
    for value in (
        "frame_index=3",
        "sample_id='predict:000003'",
        "template_id='zeta'",
        "solver_path='eval_adaptive'",
        "prediction_stage='structure_domain_preflight'",
        "predictor_reason_code='UNSUPPORTED_SPECIES'",
    ):
        assert value in formatted
