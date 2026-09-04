"""RTX 3090 CUDA delta gate executed solely from an installed project wheel."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import pytest

from test_wheel_installation import (
    InstalledWheelEnvironment,
    build_installed_wheel_environment,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIRE_CUDA = os.environ.get("REFSITE_REQUIRE_CUDA") == "1"


def _terminal_json(result: subprocess.CompletedProcess[str]) -> Mapping[str, Any]:
    assert result.returncode == 0
    assert result.stdout.endswith("\n")
    assert result.stdout.count("\n") == 1
    value = json.loads(result.stdout)
    assert isinstance(value, Mapping)
    assert "Traceback" not in result.stderr
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _critical_run_bytes(run: Path) -> dict[str, bytes]:
    names = (
        "run_status.json",
        "metrics.jsonl",
        "checkpoints/epoch_000000.pt",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
    )
    return {name: (run / name).read_bytes() for name in names}


def _json_numeric_difference(left: Any, right: Any, *, path: str = "root") -> float:
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping) and list(left) == list(right), path
        return max(
            (
                _json_numeric_difference(left[key], right[key], path=f"{path}.{key}")
                for key in left
            ),
            default=0.0,
        )
    if isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        return max(
            (
                _json_numeric_difference(item, other, path=f"{path}[{index}]")
                for index, (item, other) in enumerate(zip(left, right))
            ),
            default=0.0,
        )
    if type(left) is float:
        assert type(right) is float, path
        return abs(left - right)
    assert type(left) is type(right) and left == right, path
    return 0.0


def _copy_support(installed: InstalledWheelEnvironment, name: str) -> Path:
    source = Path(__file__).with_name(name)
    target = installed.root / "work" / name
    shutil.copy2(source, target)
    return target


def _sync(installed: InstalledWheelEnvironment) -> None:
    synchronized = installed.run(
        (
            installed.python,
            "-c",
            "import torch; assert torch.cuda.device_count() == 1; "
            "torch.cuda.synchronize(0)",
        )
    )
    assert synchronized.stdout == ""


def _cuda_or_fail(message: str) -> None:
    if REQUIRE_CUDA:
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def installed_cuda_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> InstalledWheelEnvironment:
    root = tmp_path_factory.mktemp("refsite-mlip-cuda-wheel-")
    installed = build_installed_wheel_environment(
        root,
        repository_root=REPOSITORY_ROOT,
        cuda_visible_devices="0",
    )
    hardware = installed.run(
        (
            installed.python,
            "-c",
            "import json, pathlib, sys, torch, refsite_mlip; "
            "ok=torch.cuda.is_available(); "
            "p=torch.cuda.get_device_properties(0) if ok else None; "
            "print(json.dumps({'available':ok,'count':torch.cuda.device_count(),"
            "'name':torch.cuda.get_device_name(0) if ok else None,"
            "'capability':[p.major,p.minor] if p else None,"
            "'module':str(pathlib.Path(refsite_mlip.__file__).resolve()),"
            "'sys_path':[str(pathlib.Path(x or '.').resolve()) for x in sys.path]},"
            "sort_keys=True))",
        ),
        check=False,
    )
    if hardware.returncode != 0:
        _cuda_or_fail(
            "installed-wheel CUDA hardware probe failed: " + hardware.stderr
        )
    metadata = _terminal_json(hardware)
    if not metadata["available"] or metadata["count"] != 1:
        _cuda_or_fail(f"expected one visible CUDA device, observed {metadata}")
    if metadata["name"] != "NVIDIA GeForce RTX 3090":
        _cuda_or_fail(f"expected NVIDIA GeForce RTX 3090, observed {metadata['name']}")
    assert metadata["capability"] == [8, 6]
    module = Path(metadata["module"])
    assert "site-packages" in module.parts
    assert not module.is_relative_to(REPOSITORY_ROOT / "src")
    assert str(REPOSITORY_ROOT) not in metadata["sys_path"]
    assert str(REPOSITORY_ROOT / "src") not in metadata["sys_path"]
    assert installed.environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert installed.environment["CUDA_LAUNCH_BLOCKING"] == "1"
    return installed


@dataclass(frozen=True)
class _CudaCase:
    dtype: str
    manifest: Mapping[str, Any]
    split_run: Path
    continuous_run: Path
    report: Mapping[str, Any]
    hidden_report: Mapping[str, Any]
    hardware: Mapping[str, Any]
    repeated_evaluate_max_abs_error: float


def _run_case(installed: InstalledWheelEnvironment, dtype: str) -> _CudaCase:
    fixture_script = _copy_support(installed, "installed_workflow_fixture.py")
    _copy_support(installed, "installed_workflow_probe.py")
    probe_script = _copy_support(installed, "installed_cuda_workflow_probe.py")
    hidden_probe_script = _copy_support(
        installed, "installed_cuda_hidden_restore_probe.py"
    )
    fixture_root = installed.root / f"cuda-fixture-{dtype}"
    generated = installed.run(
        (
            installed.python,
            fixture_script,
            fixture_root,
            "--transport-backend",
            "edge_list",
            "--candidate-backend",
            "blocked",
        )
    )
    manifest = _terminal_json(generated)
    assert manifest["transport_backend"] == "edge_list"
    assert manifest["candidate_backend"] == "blocked"
    assert set(manifest["template_site_counts"].values()) == {8, 16}
    assert {item["K"] for item in manifest["train_structures"]} == {0, 1}
    assert {item["K"] for item in manifest["validation_structures"]} == {0, 1}

    split = manifest["cases"]["split"]
    continuous = manifest["cases"]["continuous"]
    split_config = str(split["config"])
    continuous_config = str(continuous["config"])
    split_run = Path(split["output_directory"])
    continuous_run = Path(continuous["output_directory"])
    overrides = ("--device", "cuda:0", "--dtype", dtype, "--json")

    validate = installed.run_console(
        "validate-train-config", split_config, *overrides
    )
    dry_run = installed.run_module(
        "train", "--config", split_config, "--dry-run", *overrides
    )
    assert _terminal_json(validate) == _terminal_json(dry_run)
    assert not split_run.exists()

    trained = installed.run_console(
        "train", "--config", split_config, *overrides
    )
    trained_report = _terminal_json(trained)
    assert trained_report["status"] == "completed"
    assert "Epoch 001/1" in trained.stderr
    _sync(installed)
    required = {
        "resolved_config.json",
        "preflight.json",
        "data_manifest.json",
        "initial_bundle.pt",
        "run_status.json",
        "metrics.jsonl",
        "checkpoints/epoch_000000.pt",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
    }
    assert all((split_run / name).is_file() for name in required)
    resolved = json.loads((split_run / "resolved_config.json").read_text())
    assert resolved["runtime"] == {
        "device": "cuda:0",
        "dtype": dtype,
        "seed": manifest["training_seed"],
    }
    epoch_zero = (split_run / "checkpoints/epoch_000000.pt").read_bytes()
    best_zero = (split_run / "checkpoints/best.pt").read_bytes()
    journal_prefix = (split_run / "metrics.jsonl").read_bytes()

    resumed = installed.run_module(
        "resume", str(split_run), "--max-epochs", "2", "--json"
    )
    resumed_report = _terminal_json(resumed)
    assert resumed_report["status"] == "completed"
    assert "Epoch 002/2" in resumed.stderr and "Epoch 001/2" not in resumed.stderr
    _sync(installed)
    assert (split_run / "checkpoints/epoch_000001.pt").is_file()
    assert (split_run / "checkpoints/epoch_000000.pt").read_bytes() == epoch_zero
    assert (split_run / "checkpoints/best.pt").read_bytes() == best_zero
    resumed_journal = (split_run / "metrics.jsonl").read_bytes()
    assert resumed_journal.startswith(journal_prefix)
    journal_events = [json.loads(line) for line in resumed_journal.splitlines()]
    assert [event["epoch_index"] for event in journal_events] == [0, 1]
    assert [event["global_step_end"] for event in journal_events] == [1, 2]
    status = json.loads((split_run / "run_status.json").read_text())
    assert status["status"] == "completed"
    assert status["completed_epochs"] == 2 and status["global_step"] == 2

    continuous_result = installed.run_console(
        "train", "--config", continuous_config, "--quiet", *overrides
    )
    assert _terminal_json(continuous_result)["status"] == "completed"
    assert "Epoch " not in continuous_result.stderr
    _sync(installed)

    best_bundle = installed.root / "work" / f"best-{dtype}.pt"
    latest_bundle = installed.root / "work" / f"latest-{dtype}.pt"
    best_export = installed.run_console(
        "export-bundle",
        str(split_run),
        "--source",
        "best",
        "--output",
        best_bundle,
        "--json",
    )
    latest_export = installed.run_module(
        "export-bundle",
        str(split_run),
        "--source",
        "latest",
        "--output",
        latest_bundle,
        "--json",
    )
    assert _terminal_json(best_export)["source"]["epoch"] == 0
    assert _terminal_json(latest_export)["source"]["epoch"] == 1
    inspect_best = _terminal_json(
        installed.run_console("inspect-bundle", best_bundle, "--json")
    )
    inspect_latest = _terminal_json(
        installed.run_module("inspect-bundle", latest_bundle, "--json")
    )
    assert inspect_best["schema_version"] == inspect_latest["schema_version"]
    assert inspect_best["template_ids"] == inspect_latest["template_ids"]

    input_path = Path(split["mixed_labeled"])
    predictions = installed.root / "work" / f"predictions-{dtype}.xyz"
    predicted = installed.run_console(
        "predict",
        "--bundle",
        latest_bundle,
        "--input",
        input_path,
        "--output",
        predictions,
        "--template-key",
        manifest["template_key"],
        "--solver",
        "train-fixed",
        "--properties",
        "energy,forces,stress",
        "--device",
        "cuda:0",
        "--dtype",
        dtype,
        "--batch-size",
        "2",
        "--json",
    )
    assert _terminal_json(predicted)["frame_count"] == 4
    _sync(installed)

    evaluation_report = installed.root / "work" / f"evaluation-{dtype}.json"
    evaluation_arguments = (
        "evaluate",
        "--bundle",
        latest_bundle,
        "--input",
        input_path,
        "--template-key",
        manifest["template_key"],
        "--solver",
        "train-fixed",
        "--terms",
        "energy,forces,stress",
        "--device",
        "cuda:0",
        "--dtype",
        dtype,
        "--batch-size",
        "2",
        "--energy-mode",
        "per-atom",
        "--energy-scale",
        "2",
        "--force-scale",
        "3",
        "--stress-scale",
        "4",
        "--energy-weight",
        "1.5",
        "--force-weight",
        "2.5",
        "--stress-weight",
        "3.5",
    )
    evaluated = installed.run_module(*evaluation_arguments, "--json")
    evaluation_payload = _terminal_json(evaluated)
    assert evaluation_payload["frame_count"] == 4
    saved_evaluation = installed.run_console(
        *evaluation_arguments, "--output", evaluation_report
    )
    assert saved_evaluation.returncode == 0 and saved_evaluation.stdout == ""
    saved_payload = json.loads(evaluation_report.read_text())
    repeated_evaluate_error = _json_numeric_difference(
        saved_payload, evaluation_payload
    )
    repeated_tolerance = 6.0e-5 if dtype == "float32" else 3.0e-10
    assert repeated_evaluate_error <= repeated_tolerance
    _sync(installed)

    probed = installed.run(
        (
            installed.python,
            probe_script,
            "--fixture-manifest",
            manifest["manifest_path"],
            "--continuous-run",
            continuous_run,
            "--split-run",
            split_run,
            "--best-bundle",
            best_bundle,
            "--latest-bundle",
            latest_bundle,
            "--predictions",
            predictions,
            "--evaluation-report",
            evaluation_report,
            "--dtype",
            dtype,
        )
    )
    report = _terminal_json(probed)
    _sync(installed)

    # A CUDA-hidden process must fail before mutating the run.  The public CLI
    # provides the structured device error; the exact restore precondition is
    # separately exercised against the same weights-only checkpoint.
    before_hidden = _critical_run_bytes(split_run)
    hidden_environment = dict(installed.environment)
    hidden_environment["CUDA_VISIBLE_DEVICES"] = ""
    hidden_environment.pop("CUDA_LAUNCH_BLOCKING", None)
    hidden = replace(installed, environment=hidden_environment)
    hidden_cli = hidden.run_module(
        "resume",
        str(split_run),
        "--max-epochs",
        "3",
        "--dry-run",
        "--json",
        check=False,
    )
    assert hidden_cli.returncode == 1
    assert hidden_cli.stdout == ""
    assert "CUDA" in hidden_cli.stderr and "Traceback" not in hidden_cli.stderr
    assert _critical_run_bytes(split_run) == before_hidden
    hidden_report = _terminal_json(
        hidden.run((hidden.python, hidden_probe_script, split_run))
    )
    assert hidden_report["status"] == "rejected"
    assert hidden_report["cuda_device_count"] == 0
    assert hidden_report["artifacts_unchanged"]
    assert _critical_run_bytes(split_run) == before_hidden

    hardware = _terminal_json(
        installed.run(
            (
                installed.python,
                "-c",
                "import json, torch; p=torch.cuda.get_device_properties(0); "
                "print(json.dumps({'torch':torch.__version__,"
                "'cuda_runtime':torch.version.cuda,'device_count':torch.cuda.device_count(),"
                "'name':torch.cuda.get_device_name(0),"
                "'capability':[p.major,p.minor],"
                "'memory_bytes':p.total_memory},sort_keys=True))",
            )
        )
    )
    return _CudaCase(
        dtype=dtype,
        manifest=manifest,
        split_run=split_run,
        continuous_run=continuous_run,
        report=report,
        hidden_report=hidden_report,
        hardware=hardware,
        repeated_evaluate_max_abs_error=repeated_evaluate_error,
    )


@pytest.fixture(scope="module")
def installed_cuda_cases(
    installed_cuda_environment: InstalledWheelEnvironment,
) -> Mapping[str, _CudaCase]:
    console_version = installed_cuda_environment.run_console("version")
    module_version = installed_cuda_environment.run_module("version")
    assert console_version.stdout == module_version.stdout
    return {
        dtype: _run_case(installed_cuda_environment, dtype)
        for dtype in ("float32", "float64")
    }


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_installed_rtx3090_cuda_workflow(
    installed_cuda_cases: Mapping[str, _CudaCase], dtype: str
) -> None:
    case = installed_cuda_cases[dtype]
    report = case.report
    assert report["schema_version"] == "refsite_installed_cuda_workflow_probe_v1"
    assert report["status"] == "passed"
    assert report["device"] == {
        "capability": [8, 6],
        "count": 1,
        "name": "NVIDIA GeForce RTX 3090",
        "requested": "cuda:0",
    }
    assert report["dtype"] == dtype
    assert report["trajectory"]["restore_exact"]
    assert report["prediction"]["adaptive"]["fallback_used"] is False
    assert report["prediction"]["adaptive"]["dense_plan_materialized"] is False
    assert report["weights_only_calls"] and report["safe_global_unchanged"]
    assert case.hidden_report["weights_only_calls"]


def test_installed_cuda_float32_float64_matrix_completed_without_skip(
    installed_cuda_cases: Mapping[str, _CudaCase],
) -> None:
    assert set(installed_cuda_cases) == {"float32", "float64"}
    for dtype, case in installed_cuda_cases.items():
        assert case.hardware["device_count"] == 1
        assert case.hardware["name"] == "NVIDIA GeForce RTX 3090"
        assert case.hardware["capability"] == [8, 6]
        assert case.report["trajectory"]["runtime_dtype"] == dtype
        assert case.report["prediction"]["q_mass_max_abs_error"] >= 0.0
        assert case.repeated_evaluate_max_abs_error >= 0.0
    cross_dtype = installed_cuda_cases["float64"].report["prediction"][
        "cross_dtype_same_state_error"
    ]
    assert cross_dtype is not None
    assert cross_dtype["first_mismatch"] in {
        None,
        "energy",
        "forces",
        "stress",
        "stress_voigt",
    }
    assert all(
        value >= 0.0
        for output in cross_dtype["outputs"].values()
        for value in output.values()
    )
