from __future__ import annotations

import importlib
import json

import pytest

from refsite_mlip.cli.errors import CLIError, format_cli_error
from refsite_mlip.cli.export_bundle import (
    ExportBundleConfig,
    export_bundle,
    render_export_bundle_human,
    render_export_bundle_json,
)
from refsite_mlip.cli.main import build_parser, main
from refsite_mlip.training import TrainingRunDirectory


def _report(*, dry_run: bool = False) -> dict:
    return {
        "schema_version": "refsite_export_bundle_result_v1",
        "status": "dry_run_ready" if dry_run else "completed",
        "dry_run": dry_run,
        "output_written": not dry_run,
        "run_directory": "/runtime/run",
        "path_kind": "runtime_location_not_semantic_fingerprint",
        "source": {
            "kind": "best",
            "epoch": 2,
            "global_step": 9,
            "selection_monitor": "total_loss",
            "selection_mode": "min",
            "monitored_metric": 0.25,
        },
        "bundle_sha256": "a" * 64,
        "architecture_fingerprint": "b" * 64,
        "parent_initial_bundle_sha256": "c" * 64,
        "training_config_sha256": "d" * 64,
        "train_semantic_digest": "e" * 64,
        "validation_semantic_digest": "f" * 64,
        "template_ids": ["alpha", "zeta"],
        "template_fingerprints": {"alpha": "1" * 64, "zeta": "2" * 64},
        "species_vocabulary": [6, 41],
        "state": {
            "parameter_tensor_count": 2,
            "parameter_count": 10,
            "parameter_bytes": 80,
            "buffer_tensor_count": 1,
            "buffer_count": 3,
            "buffer_bytes": 24,
            "state_tensor_count": 3,
            "total_bytes": 104,
        },
        "radii": {
            "config_fingerprint": "3" * 64,
            "user": {"r_ot": 4.0, "r_mp": 3.0},
            "advanced": {
                "ot_switch_width": 0.5,
                "ot_skin": 0.2,
                "mp_skin": 0.5,
            },
            "derived": {
                "r_on_ot": 3.5,
                "r_off_ot": 4.0,
                "r_candidate_ot": 4.2,
                "r_mp": 3.0,
                "r_candidate_mp": 3.5,
            },
        },
        "output_path": "/runtime/best-model.pt",
        "excluded_state": [
            "dataset",
            "optimizer",
            "rng",
            "scheduler",
            "selection",
            "training_history",
        ],
        "message": "optimizer/training state excluded",
    }


def test_parser_contract_and_usage_errors():
    parser = build_parser()
    args = parser.parse_args(
        [
            "export-bundle",
            "run-output",
            "--source",
            "latest",
            "--output",
            "model.pt",
            "--initial-bundle",
            "moved.pt",
            "--dry-run",
            "--overwrite",
            "--json",
        ]
    )
    assert args.command == "export-bundle"
    assert args.source == "latest"
    assert args.output_path == "model.pt"
    assert args.initial_bundle_path == "moved.pt"
    assert args.dry_run and args.overwrite and args.json_output
    with pytest.raises(SystemExit) as missing_source:
        parser.parse_args(["export-bundle", "run", "--output", "out.pt"])
    assert missing_source.value.code == 2
    with pytest.raises(SystemExit) as arbitrary_source:
        parser.parse_args(
            ["export-bundle", "run", "--source", "epoch_1", "--output", "out.pt"]
        )
    assert arbitrary_source.value.code == 2


def test_export_config_is_strict():
    config = ExportBundleConfig("run", "best", "out.pt", dry_run=True)
    assert config.source == "best" and config.dry_run
    with pytest.raises(ValueError, match="best.*latest"):
        ExportBundleConfig("run", "epoch", "out.pt")
    with pytest.raises(TypeError, match="dry_run"):
        ExportBundleConfig("run", "best", "out.pt", dry_run=1)
    with pytest.raises(ValueError, match="output_path"):
        ExportBundleConfig("run", "best", "")


def test_json_and_human_rendering_are_deterministic_and_explicit():
    report = _report()
    reversed_report = dict(reversed(tuple(report.items())))
    first = render_export_bundle_json(report)
    second = render_export_bundle_json(reversed_report)
    assert first == second
    assert json.loads(first) == report
    assert "NaN" not in first and "Infinity" not in first
    human = render_export_bundle_human(report)
    assert "Source: best (epoch 2, global step 9)" in human
    assert "Bundle SHA-256:" in human
    assert "Template IDs: alpha, zeta" in human
    assert "Optimizer/training state excluded." in human
    dry = render_export_bundle_human(_report(dry_run=True))
    assert "ready (dry run)" in dry
    assert "No output file or run-directory file was changed." in dry


def test_cli_dispatch_json_and_human(monkeypatch, capsys):
    module = importlib.import_module("refsite_mlip.cli.export_bundle")
    calls = []

    def fake_export(config):
        calls.append(config)
        return _report(dry_run=config.dry_run)

    monkeypatch.setattr(module, "export_bundle", fake_export)
    argv = [
        "export-bundle",
        "run-output",
        "--source",
        "best",
        "--output",
        "best-model.pt",
        "--dry-run",
        "--json",
    ]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "dry_run_ready"
    assert calls[0] == ExportBundleConfig(
        "run-output", "best", "best-model.pt", dry_run=True
    )
    assert main(argv[:-1]) == 0
    captured = capsys.readouterr()
    assert "portable bundle export" in captured.out
    assert captured.err == ""


def test_structured_source_checkpoint_error_context():
    error = CLIError(
        "CHECKPOINT_MODEL_STATE_DTYPE_MISMATCH",
        "checkpoint state is incompatible",
        stage="export.model_state.validate",
        path="run/checkpoints/best.pt",
        source_kind="best",
        checkpoint_stage="model_state_dtype",
        epoch_index=3,
        global_step=12,
        bundle_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        template_id="alpha",
        template_fingerprint="c" * 64,
        underlying_reason_code="STATE_DTYPE_MISMATCH",
    )
    rendered = format_cli_error(error)
    for text in (
        "source_kind='best'",
        "checkpoint_stage='model_state_dtype'",
        "epoch_index=3",
        "global_step=12",
        "template_id='alpha'",
        "underlying_reason_code='STATE_DTYPE_MISMATCH'",
    ):
        assert text in rendered
    assert error.to_dict()["bundle_fingerprint"] == "a" * 64


def test_runtime_failure_writes_only_stderr(capsys):
    assert (
        main(
            [
                "export-bundle",
                "missing-run",
                "--source",
                "latest",
                "--output",
                "out.pt",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "source_kind='latest'" in captured.err
    assert "Traceback" not in captured.err


def test_export_checks_active_common_run_lock_before_startup_metadata(tmp_path):
    root = tmp_path / "scratch-run"
    directory = TrainingRunDirectory.create(root)
    lock = directory.acquire_resume_lock()
    try:
        # Fresh scratch training acquires the common lock immediately after
        # creating the directory, before resolved_config/preflight/status are
        # guaranteed to exist.  Export must still report active ownership,
        # rather than a misleading missing-metadata failure.
        with pytest.raises(CLIError) as caught:
            export_bundle(
                root,
                source="latest",
                output_path=tmp_path / "model.pt",
                dry_run=True,
            )
        assert caught.value.reason_code == "RESUME_LOCK_EXISTS"
        assert caught.value.stage == "export.active_run"
        assert lock.owned
        assert directory.resume_lock_path.is_file()
    finally:
        lock.release()
