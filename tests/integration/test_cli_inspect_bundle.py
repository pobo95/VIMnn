from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import refsite_mlip.data.reference_builder as reference_builder_module
import refsite_mlip.graph as graph_module
import refsite_mlip.models.bundle as bundle_module
import refsite_mlip.phase.stabilizer as stabilizer_module
from refsite_mlip.cli.inspect_bundle import inspect_bundle, render_json
from refsite_mlip.models import (
    ReferenceSitePotential,
    capture_reference_site_model_bundle,
    save_reference_site_model_bundle,
)

from test_model_bundle_runtime import _capture_case


_ROOT = Path(__file__).resolve().parents[2]


def _typed_crystal_data():
    dtype = torch.float64
    return {
        "cell": torch.tensor(
            [[4.1, 0.2, -0.1], [0.4, 3.7, 0.3], [-0.2, 0.5, 3.5]],
            dtype=dtype,
        ),
        "origin": torch.tensor([0.73, -0.41, 0.29], dtype=dtype),
        "sites": torch.tensor(
            [
                [0.03, 0.07, 0.11],
                [0.31, 0.19, 0.43],
                [0.57, 0.37, 0.23],
                [0.79, 0.71, 0.61],
                [0.17, 0.83, 0.47],
                [0.68, 0.12, 0.88],
            ],
            dtype=dtype,
        ),
        "site_types": torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
        "phase": torch.tensor([0.173, -0.219, 0.137], dtype=dtype),
        "positions": torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [2.0, 1.1, 1.4],
                [3.0, 1.7, 0.8],
                [3.8, 2.9, 2.2],
                [1.6, 3.0, 1.7],
                [3.2, 0.5, 3.1],
            ],
            dtype=dtype,
        ),
        "atom_weights": torch.eye(2, dtype=dtype).repeat(3, 1),
        "site_weights": torch.eye(2, dtype=dtype).repeat(3, 1),
        "channel_weights": torch.tensor([1.0, 1.3], dtype=dtype),
        "modes": torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]],
            dtype=torch.long,
        ),
        "mode_weights": torch.tensor([1.0, 1.1, 0.9, 0.4, 0.35], dtype=dtype),
    }


@pytest.fixture(scope="module")
def bundle_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cli-bundles")
    data = _typed_crystal_data()
    _, model, _, _, _, _, policies, mixed = _capture_case(data)
    by_id = {binding.template_id: binding for binding in mixed.template_bindings}
    default = by_id[mixed.default_template_id]

    def capture(*, policy, provenance):
        return capture_reference_site_model_bundle(
            model=model,
            structural_artifacts={
                default.template_id: default.structural_artifact
            },
            phase_specifications={
                default.template_id: default.phase_specification
            },
            evaluation_policies=(
                {default.template_id: policies[default.template_id]}
                if policy
                else None
            ),
            default_template_id=default.template_id,
            provenance=provenance,
        )

    single = capture(policy=True, provenance={"purpose": "cli-single"})
    no_policy = capture(policy=False, provenance={"purpose": "cli-no-policy"})
    ordered = replace(
        single,
        provenance={"a": {"label": "stable"}, "z": [3, 2, 1]},
        bundle_fingerprint=None,
    )
    reordered = replace(
        single,
        provenance={"z": [3, 2, 1], "a": {"label": "stable"}},
        bundle_fingerprint=None,
    )
    assert ordered.bundle_fingerprint == reordered.bundle_fingerprint

    bundles = {
        "single": single,
        "mixed": mixed,
        "no_policy": no_policy,
        "ordered": ordered,
        "reordered": reordered,
    }
    paths = {}
    for name, bundle in bundles.items():
        path = directory / f"{name}.pt"
        save_reference_site_model_bundle(path, bundle)
        paths[name] = path
    return paths


def _environment():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_ROOT / "src")
    environment["PYTHONWARNINGS"] = "ignore"
    return environment


def _run_module(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "refsite_mlip", *map(str, arguments)],
        cwd=_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_console_target(*arguments):
    code = "from refsite_mlip.cli.main import main; raise SystemExit(main())"
    return subprocess.run(
        [sys.executable, "-c", code, *map(str, arguments)],
        cwd=_ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_python_module_and_console_entry_target_have_exact_parity(bundle_files):
    arguments = ("inspect-bundle", bundle_files["mixed"], "--json")
    module = _run_module(*arguments)
    console = _run_console_target(*arguments)
    assert module.returncode == console.returncode == 0
    assert module.stdout == console.stdout
    assert module.stderr == console.stderr == ""


def test_human_json_single_mixed_and_policy_presence(bundle_files):
    single_json = _run_module("inspect-bundle", bundle_files["single"], "--json")
    mixed_json = _run_module("inspect-bundle", bundle_files["mixed"], "--json")
    no_policy_json = _run_module(
        "inspect-bundle", bundle_files["no_policy"], "--json"
    )
    human = _run_module("inspect-bundle", bundle_files["mixed"])
    assert all(
        result.returncode == 0
        for result in (single_json, mixed_json, no_policy_json, human)
    )

    single = json.loads(single_json.stdout)
    mixed = json.loads(mixed_json.stdout)
    no_policy = json.loads(no_policy_json.stdout)
    assert len(single["template_ids"]) == 1
    assert mixed["template_ids"] == ["alpha", "zeta"]
    assert single["templates"][single["default_template_id"]][
        "evaluation_policy_present"
    ]
    assert not no_policy["templates"][no_policy["default_template_id"]][
        "evaluation_policy_present"
    ]
    assert mixed["conventions"]["stress_voigt_order"] == [
        "xx",
        "yy",
        "zz",
        "yz",
        "xz",
        "xy",
    ]
    assert mixed["conventions"]["stress_sign"] == "tensile_positive"
    assert "Evaluation policy present: yes" in human.stdout
    assert human.stdout.index("  alpha\n") < human.stdout.index("  zeta\n")
    assert str(bundle_files["mixed"].resolve()) not in mixed_json.stdout
    assert str(bundle_files["mixed"].resolve()) not in human.stdout


def test_json_is_byte_stable_across_runs_and_mapping_insertion_order(bundle_files):
    first = _run_module("inspect-bundle", bundle_files["ordered"], "--json")
    repeated = _run_module("inspect-bundle", bundle_files["ordered"], "--json")
    reordered = _run_module("inspect-bundle", bundle_files["reordered"], "--json")
    assert first.returncode == repeated.returncode == reordered.returncode == 0
    assert first.stdout == repeated.stdout == reordered.stdout


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("truncated", "SAFE_LOAD_FAILURE"),
        ("schema", "UNSUPPORTED_SCHEMA"),
        ("fingerprint", "BUNDLE_FINGERPRINT_MISMATCH"),
        ("state_key", "INVALID_STATE_KEYS"),
    ],
)
def test_corruption_errors_preserve_loader_reason_and_stage(
    bundle_files, tmp_path, mutation, reason
):
    source = bundle_files["single"]
    target = tmp_path / f"{mutation}.pt"
    if mutation == "truncated":
        target.write_bytes(source.read_bytes()[:31])
    else:
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if mutation == "schema":
            payload["schema_version"] = "future_bundle_v99"
        elif mutation == "fingerprint":
            payload["bundle_fingerprint"] = "0" * 64
        else:
            key = payload["payload"]["model_state_keys"][0]
            payload["payload"]["model_state"].pop(key)
        torch.save(payload, target)

    result = _run_module("inspect-bundle", target, "--json")
    assert result.returncode == 1
    assert result.stdout == ""
    assert str(target) in result.stderr
    assert f"reason='{reason}'" in result.stderr
    assert "stage=" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_file_directory_and_debug_traceback_contract(bundle_files, tmp_path):
    missing = tmp_path / "missing.pt"
    missing_result = _run_module("inspect-bundle", missing)
    directory_result = _run_module("inspect-bundle", tmp_path)
    debug_result = _run_module(
        "inspect-bundle", bundle_files["single"].parent / "absent.pt", "--debug"
    )
    assert missing_result.returncode == directory_result.returncode == 1
    assert missing_result.stdout == directory_result.stdout == ""
    assert "reason='BUNDLE_NOT_FOUND'" in missing_result.stderr
    assert "reason='INVALID_BUNDLE_PATH'" in directory_result.stderr
    assert debug_result.returncode == 1
    assert debug_result.stdout == ""
    assert "Traceback" in debug_result.stderr
    assert "BUNDLE_NOT_FOUND" in debug_result.stderr


def test_inspection_is_cpu_only_inference_free_builder_free_and_transactional(
    bundle_files, monkeypatch
):
    path = bundle_files["mixed"]
    bytes_before = path.read_bytes()
    rng_before = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("inference/builder/CUDA query must not run")

    monkeypatch.setattr(
        reference_builder_module, "build_reference_template_from_atoms", forbidden
    )
    monkeypatch.setattr(
        reference_builder_module, "canonicalize_reference_atoms", forbidden
    )
    monkeypatch.setattr(graph_module, "build_reference_graph_topology", forbidden)
    monkeypatch.setattr(stabilizer_module, "find_typed_stabilizer", forbidden)
    monkeypatch.setattr(ReferenceSitePotential, "forward", forbidden)
    monkeypatch.setattr(
        bundle_module, "instantiate_reference_site_model_bundle", forbidden
    )
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)

    report = inspect_bundle(path)
    encoded = render_json(report)
    assert json.loads(encoded)["template_ids"] == ["alpha", "zeta"]
    assert path.read_bytes() == bytes_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert "model_state_keys" not in encoded
    assert '"model_state"' not in encoded
