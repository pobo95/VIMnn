"""Deep runtime checks for the repository-external installed-wheel workflow.

This support program is copied beside the generated fixture and executed with
the virtual environment's Python.  It deliberately imports no repository test
module, so every ``refsite_mlip`` object below comes from the installed wheel.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from ase.io import read

import refsite_mlip
from refsite_mlip.cli.evaluate import (
    ExtXYZEvaluationConfig,
    _full_prediction,
    _physical_metrics,
    _prepare_labeled_samples,
)
from refsite_mlip.cli.predict import _prepare_samples
from refsite_mlip.data import StructureSample, collate_structure_samples
from refsite_mlip.inference import load_reference_site_predictor
from refsite_mlip.interfaces import ReferenceSiteASECalculator
from refsite_mlip.models import load_reference_site_model_bundle
from refsite_mlip.training import (
    CommittedEpochMetrics,
    FitEpochRecord,
    committed_epoch_metrics_from_record,
    committed_epoch_provenance_from_checkpoint_metadata,
    compute_potential_loss,
    load_training_checkpoint,
)
from refsite_mlip.training.resume import (
    _numpy_state_from_safe,
    _python_state_from_safe,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED


def _strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"nonfinite JSON value {value!r} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=nonfinite,
    )


def _frames(path: Path) -> tuple[Any, ...]:
    loaded = read(path, index=":", format="extxyz")
    return tuple(loaded if isinstance(loaded, list) else [loaded])


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.device.type == right.device.type
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(
        left, (str, bytes, bytearray)
    ):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes, bytearray))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _assert_tree_equal(left: Any, right: Any, *, name: str) -> None:
    if not _tree_equal(left, right):
        raise AssertionError(f"{name} are not exactly equal")


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.all(torch.isfinite(value)))
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return all(_finite_tree(item) for item in value)
    if type(value) is float:
        return math.isfinite(value)
    return True


def _assert_no_presentation_time(value: Any) -> None:
    forbidden = {
        "elapsed",
        "elapsed_seconds",
        "eta",
        "eta_seconds",
        "timestamp",
        "wall_clock",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden:
                raise AssertionError(f"presentation-only field {key!r} was persisted")
            _assert_no_presentation_time(item)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _assert_no_presentation_time(item)


def _journal(path: Path) -> tuple[CommittedEpochMetrics, ...]:
    encoded = path.read_bytes()
    if not encoded.endswith(b"\n"):
        raise AssertionError("metrics journal must end in one complete line")
    result = []
    for line in encoded.splitlines():
        payload = json.loads(
            line.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite journal value {value}")
            ),
        )
        event = CommittedEpochMetrics.from_dict(payload)
        if json.dumps(
            event.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") != line:
            raise AssertionError("metrics journal line is not canonical JSON")
        result.append(event)
    return tuple(result)


def _validate_journal_against_checkpoint(
    run: Path,
    checkpoint: Any,
    initial_bundle_fingerprint: str,
) -> tuple[CommittedEpochMetrics, ...]:
    actual = _journal(run / "metrics.jsonl")
    if checkpoint.fit_history is None:
        raise AssertionError("terminal checkpoint is missing full fit history")
    provenance = committed_epoch_provenance_from_checkpoint_metadata(
        checkpoint.metadata,
        initial_bundle_fingerprint=initial_bundle_fingerprint,
    )
    selection_mode = checkpoint.metadata.resolved_configuration[
        "model_selection"
    ]["mode"]
    records = tuple(
        FitEpochRecord.from_dict(item) for item in checkpoint.fit_history
    )
    expected = tuple(
        committed_epoch_metrics_from_record(
            record,
            None,
            selection_mode=selection_mode,
            provenance=provenance,
        )
        for record in records
    )
    if actual != expected:
        raise AssertionError("journal events differ from checkpoint FitEpochRecord history")
    return actual


def _rng_draws(checkpoint: Any) -> dict[str, Any]:
    before = (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
    )
    try:
        random.setstate(_python_state_from_safe(checkpoint.python_rng_state))
        python_draw = random.random()
        np.random.set_state(_numpy_state_from_safe(checkpoint.numpy_rng_state))
        numpy_draw = float(np.random.random())
        torch.set_rng_state(checkpoint.torch_cpu_rng_state)
        torch_draw = torch.rand(4, dtype=torch.float64).tolist()
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
    return {
        "numpy": numpy_draw,
        "python": python_draw,
        "torch_cpu": torch_draw,
    }


def _state_parity(
    continuous_run: Path,
    split_run: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    continuous = load_training_checkpoint(
        continuous_run / "checkpoints" / "latest.pt"
    )
    resumed = load_training_checkpoint(split_run / "checkpoints" / "latest.pt")
    for name in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ):
        _assert_tree_equal(
            getattr(continuous, name), getattr(resumed, name), name=name
        )
    if continuous.selection_state != resumed.selection_state:
        raise AssertionError("continuous and resumed selection states differ")
    if continuous.progress != resumed.progress:
        raise AssertionError("continuous and resumed FitProgress differ")
    _assert_tree_equal(
        continuous.fit_history, resumed.fit_history, name="fit histories"
    )
    _assert_tree_equal(
        continuous.python_rng_state,
        resumed.python_rng_state,
        name="Python RNG states",
    )
    _assert_tree_equal(
        continuous.numpy_rng_state,
        resumed.numpy_rng_state,
        name="NumPy RNG states",
    )
    _assert_tree_equal(
        continuous.torch_cpu_rng_state,
        resumed.torch_cpu_rng_state,
        name="Torch CPU RNG states",
    )
    if (
        continuous.cuda_device_count != resumed.cuda_device_count
        or resumed.cuda_device_count != 0
    ):
        raise AssertionError("10C-1 wheel subprocess unexpectedly observed CUDA")
    if (continuous_run / "metrics.jsonl").read_bytes() != (
        split_run / "metrics.jsonl"
    ).read_bytes():
        raise AssertionError("continuous and resumed canonical journals differ")
    continuous_status = _strict_json(continuous_run / "run_status.json")
    resumed_status = _strict_json(split_run / "run_status.json")
    if continuous_status["fit_result"] != resumed_status["fit_result"]:
        raise AssertionError("continuous and resumed FitResult values differ")
    draws = _rng_draws(continuous)
    if draws != _rng_draws(resumed):
        raise AssertionError("continuous and resumed next RNG draws differ")
    return continuous, resumed, draws


def _bundle_parity(
    split_run: Path,
    best_bundle_path: Path,
    latest_bundle_path: Path,
) -> tuple[Any, dict[str, Any]]:
    initial = load_reference_site_model_bundle(split_run / "initial_bundle.pt")
    best_checkpoint = load_training_checkpoint(split_run / "checkpoints" / "best.pt")
    latest_checkpoint = load_training_checkpoint(
        split_run / "checkpoints" / "latest.pt"
    )
    best = load_reference_site_model_bundle(best_bundle_path)
    latest = load_reference_site_model_bundle(latest_bundle_path)
    _assert_tree_equal(
        best.model_state, best_checkpoint.model_state_dict, name="best export state"
    )
    _assert_tree_equal(
        latest.model_state,
        latest_checkpoint.model_state_dict,
        name="latest export state",
    )
    if best.bundle_fingerprint == latest.bundle_fingerprint:
        raise AssertionError("different best/latest epochs unexpectedly share a fingerprint")
    if best.provenance["source"] != "managed_epoch_checkpoint":
        raise AssertionError("best export provenance is not alias-neutral")
    if latest.provenance["source"] != "managed_epoch_checkpoint":
        raise AssertionError("latest export provenance is not alias-neutral")
    if best.provenance["checkpoint_epoch"] != 0:
        raise AssertionError("best export does not identify epoch zero")
    if latest.provenance["checkpoint_epoch"] != 1:
        raise AssertionError("latest export does not identify epoch one")

    initial_bindings = {
        item.template_id: item for item in initial.template_bindings
    }
    for exported in (best, latest):
        if (
            exported.default_template_id != initial.default_template_id
            or exported.species_vocabulary != initial.species_vocabulary
        ):
            raise AssertionError("export changed default template or species vocabulary")
        _assert_tree_equal(
            exported.conventions, initial.conventions, name="bundle conventions"
        )
        bindings = {item.template_id: item for item in exported.template_bindings}
        if set(bindings) != set(initial_bindings):
            raise AssertionError("export changed template IDs")
        for template_id, source in initial_bindings.items():
            target = bindings[template_id]
            if source.full_template_fingerprint != target.full_template_fingerprint:
                raise AssertionError("export changed full template fingerprint")
            _assert_tree_equal(
                source.structural_artifact.to_payload(),
                target.structural_artifact.to_payload(),
                name=f"{template_id} structural artifact",
            )
            _assert_tree_equal(
                source.phase_specification.to_dict(),
                target.phase_specification.to_dict(),
                name=f"{template_id} phase specification",
            )
            left_policy = (
                None
                if source.evaluation_policy is None
                else source.evaluation_policy.to_dict()
            )
            right_policy = (
                None
                if target.evaluation_policy is None
                else target.evaluation_policy.to_dict()
            )
            _assert_tree_equal(
                left_policy, right_policy, name=f"{template_id} evaluation policy"
            )

    initial_baseline = initial.model_state["atomic_baseline"]
    if not torch.equal(initial_baseline, torch.zeros_like(initial_baseline)):
        raise AssertionError("scratch initial bundle baseline is not exact zero")
    fitted = latest.model_state["atomic_baseline"]
    if torch.equal(fitted, torch.zeros_like(fitted)):
        raise AssertionError("exported checkpoint did not preserve fitted baseline")

    forbidden = {
        "optimizer_state_dict",
        "scheduler_state_dict",
        "selection_state",
        "fit_history",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "cuda_rng_states",
        "dataset",
    }

    def keys(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return set().union(*(keys(item) for item in value))
        return set()

    if forbidden & keys(latest.to_payload()):
        raise AssertionError("portable bundle contains training-only state")
    return latest, {
        "best_epoch": int(best_checkpoint.progress.last_completed_epoch),
        "best_fingerprint": best.bundle_fingerprint,
        "latest_epoch": int(latest_checkpoint.progress.last_completed_epoch),
        "latest_fingerprint": latest.bundle_fingerprint,
    }


def _geometry_samples(
    frames: tuple[Any, ...], template_key: str
) -> tuple[StructureSample, ...]:
    return tuple(
        StructureSample(
            sample_id=f"predict:{index:06d}",
            positions=torch.tensor(frame.get_positions(), dtype=torch.float64),
            atomic_numbers=torch.tensor(
                frame.get_atomic_numbers(), dtype=torch.long
            ),
            cell=torch.tensor(frame.cell.array, dtype=torch.float64),
            pbc=torch.tensor(frame.get_pbc(), dtype=torch.bool),
            origin=torch.zeros(3, dtype=torch.float64),
            template_id=str(frame.info[template_key]),
        )
        for index, frame in enumerate(frames)
    )


def _max_error(left: Any, right: Any) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(first - second))) if first.size else 0.0


def _prediction_and_ase_parity(
    bundle_path: Path,
    input_path: Path,
    predictions_path: Path,
    template_key: str,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    source = _frames(input_path)
    written = _frames(predictions_path)
    if len(source) != len(written):
        raise AssertionError("prediction output changed frame count")
    samples = _geometry_samples(source, template_key)
    predictor = load_reference_site_predictor(
        bundle_path, device="cpu", dtype=torch.float64
    )
    direct = predictor.predict_samples(
        samples,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
    )
    maxima = {"energy": 0.0, "forces": 0.0, "stress": 0.0, "stress_tensor": 0.0}
    for index, (original, output, expected) in enumerate(
        zip(source, written, direct.structures)
    ):
        if str(original.info[template_key]) != expected.template_id:
            raise AssertionError("direct prediction changed template ordering")
        if output.info["refsite_template_id"] != expected.template_id:
            raise AssertionError("extxyz output changed template ordering")
        if output.info["refsite_solver_path"] != TRAIN_FIXED:
            raise AssertionError("extxyz output records the wrong solver")
        for left, right, name in (
            (output.numbers, original.numbers, "atomic numbers"),
            (output.pbc, original.pbc, "PBC"),
            (output.arrays["input_order"], original.arrays["input_order"], "atom order"),
        ):
            if not np.array_equal(left, right):
                raise AssertionError(f"prediction output changed {name}")
        for left, right, name in (
            (output.positions, original.positions, "positions"),
            (output.cell.array, original.cell.array, "cell"),
        ):
            if _max_error(left, right) > 5.0e-9:
                raise AssertionError(f"prediction output changed {name}")
        energy_error = abs(output.get_potential_energy() - float(expected.energy))
        free_error = abs(
            output.get_potential_energy(force_consistent=True) - float(expected.energy)
        )
        force_error = _max_error(output.get_forces(), expected.forces.numpy())
        stress_error = _max_error(
            output.get_stress(), expected.stress_voigt.numpy()
        )
        tensor_error = _max_error(
            output.get_stress(voigt=False), expected.stress.numpy()
        )
        maxima["energy"] = max(maxima["energy"], energy_error, free_error)
        maxima["forces"] = max(maxima["forces"], force_error)
        maxima["stress"] = max(maxima["stress"], stress_error)
        maxima["stress_tensor"] = max(maxima["stress_tensor"], tensor_error)
        tensor = output.get_stress(voigt=False)
        voigt = output.get_stress()
        if not np.array_equal(voigt[[3, 4, 5]], tensor[[1, 0, 0], [2, 2, 1]]):
            raise AssertionError("ASE Voigt order is not xx yy zz yz xz xy")
        if index < 2:
            atoms = original.copy()
            atoms.calc = ReferenceSiteASECalculator(
                bundle_path,
                template_id=expected.template_id,
                device="cpu",
                dtype=torch.float64,
                solver_path=TRAIN_FIXED,
            )
            maxima.setdefault("ase_energy", 0.0)
            maxima.setdefault("ase_forces", 0.0)
            maxima.setdefault("ase_stress", 0.0)
            maxima["ase_energy"] = max(
                maxima["ase_energy"],
                abs(atoms.get_potential_energy() - float(expected.energy)),
                abs(
                    atoms.get_potential_energy(force_consistent=True)
                    - float(expected.energy)
                ),
            )
            maxima["ase_forces"] = max(
                maxima["ase_forces"],
                _max_error(atoms.get_forces(), expected.forces.numpy()),
            )
            maxima["ase_stress"] = max(
                maxima["ase_stress"],
                _max_error(atoms.get_stress(), expected.stress_voigt.numpy()),
                _max_error(atoms.get_stress(voigt=False), expected.stress.numpy()),
            )
    if any(value > 6.0e-8 for value in maxima.values()):
        raise AssertionError(f"Predictor/CLI/ASE parity exceeded tolerance: {maxima}")

    # The fixture policies were constructed from these exact templates.  One
    # pristine frame exercises the installed adaptive derivative path without
    # inventing a new branch-sensitive geometry.
    adaptive = predictor.predict_sample(
        samples[0],
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
    )
    adaptive_atoms = source[0].copy()
    adaptive_atoms.calc = ReferenceSiteASECalculator(
        bundle_path,
        template_id=samples[0].template_id,
        device="cpu",
        dtype=torch.float64,
        solver_path=EVAL_ADAPTIVE,
    )
    adaptive_errors = {
        "energy": abs(adaptive_atoms.get_potential_energy() - float(adaptive.energy)),
        "forces": _max_error(adaptive_atoms.get_forces(), adaptive.forces.numpy()),
        "stress": max(
            _max_error(adaptive_atoms.get_stress(), adaptive.stress_voigt.numpy()),
            _max_error(adaptive_atoms.get_stress(voigt=False), adaptive.stress.numpy()),
        ),
    }
    if not _finite_tree(adaptive_errors) or any(
        value > 4.0e-12 for value in adaptive_errors.values()
    ):
        raise AssertionError(f"adaptive Predictor/ASE parity failed: {adaptive_errors}")
    maxima["adaptive_predictor_ase_max"] = max(adaptive_errors.values())
    return predictor, source, maxima


def _nested_max_error(left: Any, right: Any) -> float:
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or set(left) != set(right):
            raise AssertionError("metric mappings differ")
        return max(
            (_nested_max_error(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, Sequence) and not isinstance(
        left, (str, bytes, bytearray)
    ):
        if not isinstance(right, Sequence) or len(left) != len(right):
            raise AssertionError("metric sequences differ")
        return max(
            (_nested_max_error(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            raise AssertionError("metric scalar types differ")
        return abs(float(left) - float(right))
    if left != right:
        raise AssertionError(f"metric values differ: {left!r} != {right!r}")
    return 0.0


def _evaluation_parity(
    report_path: Path,
    input_path: Path,
    bundle_path: Path,
    predictor: Any,
    frames: tuple[Any, ...],
    template_key: str,
) -> float:
    report = _strict_json(report_path)
    config = ExtXYZEvaluationConfig(
        bundle_path=str(bundle_path),
        input_path=str(input_path),
        template_key=template_key,
        solver_path=TRAIN_FIXED,
        terms=tuple(report["requested_terms"]),
        device="cpu",
        dtype=torch.float64,
        batch_size=2,
        energy_mode=report["energy_mode"],
        energy_scale=report["scales"]["energy"],
        force_scale=report["scales"]["forces"],
        stress_scale=report["scales"]["stress"],
        energy_weight=report["weights"]["energy"],
        force_weight=report["weights"]["forces"],
        stress_weight=report["weights"]["stress"],
    )
    geometry = _prepare_samples(frames, predictor, config, input_path.resolve())
    labeled = _prepare_labeled_samples(
        frames, geometry, config, input_path.resolve()
    )
    direct = predictor.predict_samples(
        geometry,
        solver_path=TRAIN_FIXED,
        compute_forces=True,
        compute_stress=True,
        return_aux=False,
        candidate_neighbor_states=None,
        return_candidate_neighbor_states=False,
    )
    prediction = _full_prediction(direct.structures, config)
    batch = collate_structure_samples(labeled, predictor.registry)
    with torch.no_grad():
        loss = compute_potential_loss(prediction, batch, config.loss_config())
    expected_loss = {
        "terms": {},
        "total_normalized": float(loss.total.detach().cpu()),
    }
    for term, attribute in (
        ("energy", "energy"),
        ("forces", "force"),
        ("stress", "stress"),
    ):
        value = getattr(loss, attribute)
        expected_loss["terms"][term] = {
            "denominator": float(value.denominator.detach().cpu()),
            "mean": float(value.mean.detach().cpu()),
            "numerator": float(value.numerator.detach().cpu()),
            "valid_count": int(value.valid_count.detach().cpu()),
            "weight": report["weights"][term],
        }
    expected_metrics = _physical_metrics(prediction, batch, config.terms)
    maximum = max(
        _nested_max_error(report["loss"], expected_loss),
        _nested_max_error(report["metrics"], expected_metrics),
    )
    if maximum > 2.0e-12:
        raise AssertionError(f"evaluate/direct metric parity exceeded tolerance: {maximum}")
    if report["template_frame_counts"] != {"alpha-111": 2, "zeta-211": 2}:
        raise AssertionError("evaluate changed frame/template ordering or counts")
    if not _finite_tree(report):
        raise AssertionError("evaluation report contains a nonfinite value")
    return maximum


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest = _strict_json(arguments.fixture_manifest)
    continuous_run = arguments.continuous_run.resolve()
    split_run = arguments.split_run.resolve()
    best_bundle_path = arguments.best_bundle.resolve()
    latest_bundle_path = arguments.latest_bundle.resolve()
    predictions_path = arguments.predictions.resolve()
    evaluation_report_path = arguments.evaluation_report.resolve()

    safe_globals_before = tuple(torch.serialization.get_safe_globals())
    original_torch_load = torch.load
    load_contract: list[bool] = []

    def audited_torch_load(*args: Any, **kwargs: Any) -> Any:
        load_contract.append(kwargs.get("weights_only") is True)
        return original_torch_load(*args, **kwargs)

    torch.load = audited_torch_load
    try:
        continuous, resumed, draws = _state_parity(continuous_run, split_run)
        continuous_initial = load_reference_site_model_bundle(
            continuous_run / "initial_bundle.pt"
        )
        split_initial = load_reference_site_model_bundle(split_run / "initial_bundle.pt")
        if continuous_initial.bundle_fingerprint != split_initial.bundle_fingerprint:
            raise AssertionError("deterministic scratch initial bundles differ")
        continuous_events = _validate_journal_against_checkpoint(
            continuous_run, continuous, continuous_initial.bundle_fingerprint
        )
        resumed_events = _validate_journal_against_checkpoint(
            split_run, resumed, split_initial.bundle_fingerprint
        )
        if continuous_events != resumed_events or len(resumed_events) != 2:
            raise AssertionError("continuous/resumed journal event history differs")
        latest, export = _bundle_parity(
            split_run, best_bundle_path, latest_bundle_path
        )
        input_path = Path(manifest["cases"]["split"]["mixed_labeled"])
        predictor, input_frames, parity = _prediction_and_ase_parity(
            latest_bundle_path,
            input_path,
            predictions_path,
            manifest["template_key"],
        )
        evaluation_error = _evaluation_parity(
            evaluation_report_path,
            input_path,
            latest_bundle_path,
            predictor,
            input_frames,
            manifest["template_key"],
        )
        for path in (
            continuous_run / "resolved_config.json",
            continuous_run / "preflight.json",
            continuous_run / "data_manifest.json",
            continuous_run / "run_status.json",
            split_run / "resolved_config.json",
            split_run / "preflight.json",
            split_run / "data_manifest.json",
            split_run / "run_status.json",
        ):
            _assert_no_presentation_time(_strict_json(path))
        _assert_no_presentation_time(continuous.to_dict())
        _assert_no_presentation_time(resumed.to_dict())
        _assert_no_presentation_time(latest.to_payload())
    finally:
        torch.load = original_torch_load

    if not load_contract or not all(load_contract):
        raise AssertionError(f"unsafe torch.load call observed: {load_contract}")
    if tuple(torch.serialization.get_safe_globals()) != safe_globals_before:
        raise AssertionError("safe-global registry changed during installed workflow probe")
    installed_module = Path(refsite_mlip.__file__).resolve()
    if "site-packages" not in installed_module.parts:
        raise AssertionError("probe did not import refsite_mlip from site-packages")
    return {
        "adaptive": {
            "executed": True,
            "predictor_ase_max_abs_error": parity[
                "adaptive_predictor_ase_max"
            ],
            "solver": "eval-adaptive",
        },
        "continuous_vs_resumed": {
            "exact": True,
            "global_step": resumed.progress.global_step,
            "history_length": len(resumed.fit_history or ()),
            "journal_events": len(resumed_events),
            "next_rng_draws": draws,
        },
        "evaluate_direct_max_abs_error": evaluation_error,
        "exports": export,
        "installed_module": str(installed_module),
        "prediction_parity_max_abs_error": parity,
        "safe_global_unchanged": True,
        "schema_version": "refsite_installed_workflow_probe_v1",
        "status": "passed",
        "torch_load_calls": len(load_contract),
        "weights_only_calls": all(load_contract),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--continuous-run", required=True, type=Path)
    parser.add_argument("--split-run", required=True, type=Path)
    parser.add_argument("--best-bundle", required=True, type=Path)
    parser.add_argument("--latest-bundle", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = _run(arguments)
    print(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
