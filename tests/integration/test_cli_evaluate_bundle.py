from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

import refsite_mlip.cli.evaluate as evaluate_module
from refsite_mlip.cli.errors import CLIError
from refsite_mlip.cli.evaluate import (
    ExtXYZEvaluationConfig,
    _full_prediction,
    _physical_metrics,
    _prepare_labeled_samples,
    evaluate_extxyz,
)
from refsite_mlip.cli.main import main
from refsite_mlip.cli.predict import _prepare_samples
from refsite_mlip.data import StructureSample, collate_structure_samples
from refsite_mlip.inference import load_reference_site_predictor
from refsite_mlip.models import (
    capture_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.training import compute_potential_loss
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from test_cli_inspect_bundle import _typed_crystal_data
from test_model_bundle_runtime import _capture_case


@pytest.fixture(scope="module")
def evaluation_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cli-evaluate-bundle")
    values = _capture_case(_typed_crystal_data())
    _, model, _, samples, _, _, _, mixed = values
    mixed_path = directory / "mixed.pt"
    save_reference_site_model_bundle(mixed_path, mixed)
    bindings = {binding.template_id: binding for binding in mixed.template_bindings}
    no_policy = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts={
            key: binding.structural_artifact for key, binding in bindings.items()
        },
        phase_specifications={
            key: binding.phase_specification for key, binding in bindings.items()
        },
        evaluation_policies=None,
        default_template_id=mixed.default_template_id,
        provenance={"purpose": "cli-evaluate-no-policy"},
    )
    no_policy_path = directory / "no-policy.pt"
    save_reference_site_model_bundle(no_policy_path, no_policy)
    return {"mixed": mixed_path, "no_policy": no_policy_path, "samples": samples}


def _atoms(sample, index: int, *, labels=("energy", "forces", "stress"), masks=False):
    atoms = Atoms(
        numbers=sample.atomic_numbers.detach().cpu().numpy(),
        positions=sample.positions.detach().cpu().numpy(),
        cell=sample.cell.detach().cpu().numpy(),
        pbc=sample.pbc.detach().cpu().numpy(),
        info={"template": sample.template_id, "source_tag": f"frame-{index}"},
    )
    results = {}
    if "energy" in labels:
        results["energy"] = 1.25 * (index + 1)
    if "forces" in labels:
        results["forces"] = (
            np.arange(len(atoms) * 3, dtype=float).reshape(len(atoms), 3) * 0.01
            + 0.1 * index
        )
    if "stress" in labels:
        results["stress"] = np.array(
            [0.01, 0.02, 0.03, 0.004, -0.005, 0.006], dtype=float
        ) + 0.001 * index
    atoms.calc = SinglePointCalculator(atoms, **results)
    if masks and "forces" in labels:
        force_mask = np.ones((len(atoms), 3), dtype=bool)
        force_mask[0, 1] = False
        atoms.arrays["force_mask"] = force_mask
    if masks and "stress" in labels:
        atoms.info["stress_mask"] = np.array(
            [True, True, False, True, False, True], dtype=bool
        )
    return atoms


def _write_input(path, samples, *, labels=("energy", "forces", "stress"), masks=False):
    frames = tuple(
        _atoms(sample, index, labels=labels, masks=masks)
        for index, sample in enumerate(samples)
    )
    write(path, frames, format="extxyz")
    return frames


def _read_all(path):
    values = read(path, index=":", format="extxyz")
    return tuple(values if isinstance(values, list) else [values])


def _direct_report_parts(path, bundle_path, config):
    frames = _read_all(path)
    predictor = load_reference_site_predictor(
        bundle_path, device=config.device, dtype=config.dtype
    )
    geometry = _prepare_samples(frames, predictor, config, path.resolve())
    labeled = _prepare_labeled_samples(frames, geometry, config, path.resolve())
    direct = predictor.predict_samples(
        geometry,
        solver_path=config.solver_path,
        compute_forces=config.compute_forces,
        compute_stress=config.compute_stress,
        return_aux=False,
        candidate_neighbor_states=None,
        return_candidate_neighbor_states=False,
    )
    prediction = _full_prediction(direct.structures, config)
    batch = collate_structure_samples(labeled, predictor.registry)
    loss = compute_potential_loss(prediction, batch, config.loss_config())
    metrics = _physical_metrics(prediction, batch, config.terms)
    return predictor, labeled, loss, metrics


def _assert_nested_close(left, right, *, tolerance=2e-12):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_close(left[key], right[key], tolerance=tolerance)
    elif isinstance(left, list):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_close(first, second, tolerance=tolerance)
    elif isinstance(left, float):
        assert left == pytest.approx(right, abs=tolerance, rel=tolerance)
    else:
        assert left == right


def test_fixed_mixed_direct_predictor_loss_metric_and_mask_parity(
    evaluation_bundle, tmp_path
):
    source = tmp_path / "mixed.xyz"
    _write_input(source, evaluation_bundle["samples"], masks=True)
    config = ExtXYZEvaluationConfig(
        bundle_path=evaluation_bundle["mixed"],
        input_path=source,
        template_key="template",
        terms=("energy", "forces", "stress"),
        energy_mode="per-atom",
        energy_scale=2.0,
        force_scale=3.0,
        stress_scale=4.0,
        energy_weight=1.5,
        force_weight=2.5,
        stress_weight=3.5,
        batch_size=2,
    )
    report = evaluate_extxyz(config)
    predictor, labeled, direct_loss, direct_metrics = _direct_report_parts(
        source, evaluation_bundle["mixed"], config
    )
    assert report["bundle_sha256"] == predictor.bundle_fingerprint
    assert report["frame_count"] == 3
    assert report["template_frame_counts"] == {"alpha": 1, "zeta": 2}
    assert report["conventions"]["stress_sign"] == "tensile_positive"
    assert report["conventions"]["stress_voigt_order"] == [
        "xx",
        "yy",
        "zz",
        "yz",
        "xz",
        "xy",
    ]
    assert report["loss"]["total_normalized"] == pytest.approx(
        float(direct_loss.total), abs=2e-12, rel=2e-12
    )
    names = {"energy": "energy", "forces": "force", "stress": "stress"}
    for term, name in names.items():
        actual = report["loss"]["terms"][term]
        expected = getattr(direct_loss, name)
        assert actual["numerator"] == pytest.approx(float(expected.numerator))
        assert actual["denominator"] == pytest.approx(float(expected.denominator))
        assert actual["mean"] == pytest.approx(float(expected.mean))
        assert actual["valid_count"] == int(expected.valid_count)
    _assert_nested_close(report["metrics"], direct_metrics)
    assert report["labels"]["forces"]["valid_count"] == sum(
        int(torch.count_nonzero(sample.force_mask)) for sample in labeled
    )
    assert report["labels"]["stress"]["valid_count"] == 12
    assert len(report["input_semantic_sha256"]) == 64
    assert str(source.resolve()) not in json.dumps(report, sort_keys=True)


def test_batch_size_split_and_frame_permutation_metric_parity(
    evaluation_bundle, tmp_path
):
    source = tmp_path / "ordered.xyz"
    frames = _write_input(source, evaluation_bundle["samples"], masks=False)
    reports = []
    for batch_size in (1, 16):
        reports.append(
            evaluate_extxyz(
                ExtXYZEvaluationConfig(
                    bundle_path=evaluation_bundle["mixed"],
                    input_path=source,
                    template_key="template",
                    terms=("energy", "forces", "stress"),
                    batch_size=batch_size,
                )
            )
        )
    for key in ("metrics", "loss", "labels", "composition", "template_frame_counts"):
        _assert_nested_close(reports[0][key], reports[1][key], tolerance=3e-12)

    permuted = tmp_path / "permuted.xyz"
    write(permuted, tuple(reversed(frames)), format="extxyz")
    permuted_report = evaluate_extxyz(
        ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=permuted,
            template_key="template",
            terms=("energy", "forces", "stress"),
            batch_size=2,
        )
    )
    for key in ("metrics", "loss", "labels", "composition", "template_frame_counts"):
        _assert_nested_close(reports[0][key], permuted_report[key], tolerance=3e-12)
    assert reports[0]["input_semantic_sha256"] != permuted_report[
        "input_semantic_sha256"
    ]


def test_adaptive_all_terms_and_energy_modes(evaluation_bundle, tmp_path):
    source = tmp_path / "adaptive.xyz"
    _write_input(source, evaluation_bundle["samples"], masks=False)
    reports = {}
    for mode in ("per-structure", "per-atom"):
        config = ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=source,
            template_key="template",
            solver_path=EVAL_ADAPTIVE,
            terms=("energy", "forces", "stress"),
            energy_mode=mode,
            batch_size=2,
        )
        reports[mode] = evaluate_extxyz(config)
    assert reports["per-structure"]["solver"] == "eval-adaptive"
    assert reports["per-atom"]["energy_mode"] == "per-atom"
    assert (
        reports["per-structure"]["loss"]["terms"]["energy"]["numerator"]
        != reports["per-atom"]["loss"]["terms"]["energy"]["numerator"]
    )
    _assert_nested_close(
        reports["per-structure"]["metrics"],
        reports["per-atom"]["metrics"],
        tolerance=2e-12,
    )


def test_template_default_exact_id_key_and_index_selection(evaluation_bundle, tmp_path):
    samples = evaluation_bundle["samples"]
    zeta_source = tmp_path / "zeta.xyz"
    zeta_frames = _write_input(zeta_source, (samples[0], samples[2]))
    for frame in zeta_frames:
        frame.info.pop("template")
    write(zeta_source, zeta_frames, format="extxyz")
    default = evaluate_extxyz(
        ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=zeta_source,
            terms=("energy",),
        )
    )
    assert default["template_frame_counts"] == {"zeta": 2}

    alpha_source = tmp_path / "alpha.xyz"
    alpha = _atoms(samples[1], 0)
    alpha.info["template"] = "wrong-info-is-overridden"
    write(alpha_source, alpha, format="extxyz")
    exact = evaluate_extxyz(
        ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=alpha_source,
            template_id="alpha",
            terms=("energy",),
        )
    )
    assert exact["template_frame_counts"] == {"alpha": 1}

    selected = evaluate_extxyz(
        ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=zeta_source,
            index="1",
            template_id="zeta",
            terms=("energy",),
        )
    )
    assert selected["frame_count"] == 1


def test_labels_do_not_enter_prediction_and_missing_counts(evaluation_bundle, tmp_path):
    samples = evaluation_bundle["samples"]
    full = tmp_path / "full.xyz"
    shifted = tmp_path / "shifted.xyz"
    full_frames = _write_input(full, samples)
    shifted_frames = []
    for index, frame in enumerate(full_frames):
        changed = frame.copy()
        changed.calc = SinglePointCalculator(
            changed,
            energy=float(frame.calc.results["energy"]) + 100.0,
            forces=np.asarray(frame.calc.results["forces"]) + 50.0,
            stress=np.asarray(frame.calc.results["stress"]) - 25.0,
        )
        shifted_frames.append(changed)
    write(shifted, shifted_frames, format="extxyz")
    base_config = dict(
        bundle_path=evaluation_bundle["mixed"],
        template_key="template",
        terms=("energy", "forces", "stress"),
    )
    first = evaluate_extxyz(ExtXYZEvaluationConfig(input_path=full, **base_config))
    second = evaluate_extxyz(
        ExtXYZEvaluationConfig(input_path=shifted, **base_config)
    )
    assert first["metrics"] != second["metrics"]

    predictor = load_reference_site_predictor(evaluation_bundle["mixed"])
    first_geometry = _prepare_samples(
        _read_all(full),
        predictor,
        ExtXYZEvaluationConfig(input_path=full, **base_config),
        full.resolve(),
    )
    second_geometry = _prepare_samples(
        _read_all(shifted),
        predictor,
        ExtXYZEvaluationConfig(input_path=shifted, **base_config),
        shifted.resolve(),
    )
    left = predictor.predict_samples(first_geometry, compute_forces=True, compute_stress=True)
    right = predictor.predict_samples(second_geometry, compute_forces=True, compute_stress=True)
    assert torch.equal(left.energy, right.energy)
    assert torch.equal(left.forces, right.forces)
    assert torch.equal(left.stress, right.stress)

    missing_source = tmp_path / "partial.xyz"
    partial = (
        _atoms(samples[0], 0, labels=("energy", "forces")),
        _atoms(samples[2], 1, labels=("energy",)),
    )
    write(missing_source, partial, format="extxyz")
    partial_report = evaluate_extxyz(
        ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=missing_source,
            template_id="zeta",
            terms=("energy", "forces"),
        )
    )
    assert partial_report["labels"]["energy"]["missing_frames"] == 0
    assert partial_report["labels"]["forces"]["missing_frames"] == 1


def test_deterministic_stdout_atomic_report_overwrite_and_symlink(
    evaluation_bundle, tmp_path, capsys
):
    source = tmp_path / "input.xyz"
    _write_input(source, evaluation_bundle["samples"], labels=("energy",))
    base = [
        "evaluate",
        "--bundle",
        str(evaluation_bundle["mixed"]),
        "--input",
        str(source),
        "--template-key",
        "template",
        "--terms",
        "energy",
        "--json",
    ]
    assert main(base) == 0
    first = capsys.readouterr()
    assert first.err == ""
    parsed = json.loads(first.out)
    assert parsed["frame_count"] == 3
    assert main(base) == 0
    second = capsys.readouterr()
    assert second.out == first.out and second.err == ""

    target = tmp_path / "report.json"
    assert main([*base, "--output", str(target)]) == 0
    written = capsys.readouterr()
    assert written.out == written.err == ""
    assert json.loads(target.read_text()) == parsed
    original = target.read_bytes()
    assert main([*base, "--output", str(target)]) == 1
    collision = capsys.readouterr()
    assert collision.out == "" and "OUTPUT_EXISTS" in collision.err
    assert target.read_bytes() == original
    assert main([*base, "--output", str(target), "--overwrite"]) == 0
    assert capsys.readouterr().out == ""

    actual = tmp_path / "actual.json"
    actual.write_text("preserve")
    link = tmp_path / "link.json"
    link.symlink_to(actual)
    assert main([*base, "--output", str(link)]) == 1
    rejected = capsys.readouterr()
    assert rejected.out == "" and "OUTPUT_SYMLINK_REJECTED" in rejected.err
    assert actual.read_text() == "preserve"


def test_single_load_read_only_rng_and_no_training_calls(
    evaluation_bundle, tmp_path, monkeypatch
):
    source = tmp_path / "input.xyz"
    _write_input(source, evaluation_bundle["samples"], labels=("energy", "forces"))
    source_bytes = source.read_bytes()
    bundle_bytes = evaluation_bundle["mixed"].read_bytes()
    rng = torch.get_rng_state().clone()
    grad_enabled = torch.is_grad_enabled()
    original_load = evaluate_module._load_predictor
    loaded = []
    model_states = []

    def counted_load(config):
        predictor = original_load(config)
        loaded.append(predictor)
        model_states.append(
            {
                key: value.detach().clone()
                for key, value in predictor.model.state_dict().items()
            }
        )
        return predictor

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("training/backward API must not be called")

    monkeypatch.setattr(evaluate_module, "_load_predictor", counted_load)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    report = evaluate_extxyz(
        ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=source,
            template_key="template",
            terms=("energy", "forces"),
            batch_size=1,
        )
    )
    assert report["frame_count"] == 3 and len(loaded) == 1
    assert torch.is_grad_enabled() is grad_enabled
    assert torch.equal(torch.get_rng_state(), rng)
    assert source.read_bytes() == source_bytes
    assert evaluation_bundle["mixed"].read_bytes() == bundle_bytes
    predictor = loaded[0]
    assert not predictor.model.training
    assert all(parameter.grad is None for parameter in predictor.model.parameters())
    for key, value in predictor.model.state_dict().items():
        assert torch.equal(value, model_states[0][key])


def test_missing_nonfinite_corrupt_policy_and_error_context(
    evaluation_bundle, tmp_path
):
    sample = evaluation_bundle["samples"][0]
    missing = tmp_path / "missing.xyz"
    write(missing, _atoms(sample, 0, labels=("energy",)), format="extxyz")
    with pytest.raises(CLIError) as caught:
        evaluate_extxyz(
            ExtXYZEvaluationConfig(
                bundle_path=evaluation_bundle["mixed"],
                input_path=missing,
                template_id="zeta",
                terms=("stress",),
            )
        )
    assert caught.value.reason_code == "NO_VALID_LABELS"
    assert caught.value.term == "stress"
    assert caught.value.frame_index == 0
    assert caught.value.sample_id == "evaluate:000000"
    assert caught.value.template_id == "zeta"

    nonfinite = tmp_path / "nonfinite.xyz"
    atoms = _atoms(sample, 0, labels=("energy",))
    atoms.calc.results["energy"] = float("nan")
    write(nonfinite, atoms, format="extxyz")
    with pytest.raises(CLIError) as caught:
        evaluate_extxyz(
            ExtXYZEvaluationConfig(
                bundle_path=evaluation_bundle["mixed"],
                input_path=nonfinite,
                template_id="zeta",
                terms=("energy",),
            )
        )
    assert caught.value.reason_code == "NONFINITE_LABEL"
    assert caught.value.term == "energy"

    malformed = tmp_path / "malformed.xyz"
    malformed.write_text("not extxyz\n")
    with pytest.raises(CLIError) as caught:
        evaluate_extxyz(
            ExtXYZEvaluationConfig(
                bundle_path=evaluation_bundle["mixed"],
                input_path=malformed,
                output_path=tmp_path / "no-report.json",
                template_id="zeta",
                terms=("energy",),
            )
        )
    assert caught.value.reason_code == "MALFORMED_EXTXYZ"

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"corrupt bundle")
    with pytest.raises(CLIError) as caught:
        evaluate_extxyz(
            ExtXYZEvaluationConfig(
                bundle_path=corrupt,
                input_path=missing,
                template_id="zeta",
                terms=("energy",),
            )
        )
    assert caught.value.predictor_reason_code

    with pytest.raises(CLIError) as caught:
        evaluate_extxyz(
            ExtXYZEvaluationConfig(
                bundle_path=evaluation_bundle["no_policy"],
                input_path=missing,
                template_id="zeta",
                terms=("energy",),
                solver_path=EVAL_ADAPTIVE,
            )
        )
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable on this node; retained for the Milestone 9D gate",
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("solver", [TRAIN_FIXED, EVAL_ADAPTIVE])
def test_cuda_focused_evaluation_smoke(
    evaluation_bundle, tmp_path, dtype, solver
):
    source = tmp_path / f"cuda-{dtype}-{solver}.xyz"
    _write_input(source, evaluation_bundle["samples"][:1])
    report = evaluate_extxyz(
        ExtXYZEvaluationConfig(
            bundle_path=evaluation_bundle["mixed"],
            input_path=source,
            template_id="zeta",
            terms=("energy", "forces", "stress"),
            solver_path=solver,
            device="cuda",
            dtype=dtype,
        )
    )
    assert report["device"] == "cuda" and report["dtype"] == dtype
    assert np.isfinite(report["loss"]["total_normalized"])
