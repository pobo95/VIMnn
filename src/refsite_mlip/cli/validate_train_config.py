"""Read-only training-run configuration preflight command support."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

from refsite_mlip.config import (
    ResolvedScratchTrainingRun,
    ResolvedTrainingRun,
    ScratchModelSourceConfig,
    TrainingRunConfigOverrides,
    TrainingRunConfigError,
    load_effective_training_run_config,
    resolve_training_run,
)
from refsite_mlip.training import (
    ScratchTrainingPreparation,
    prepare_scratch_training_run,
)

from .errors import CLIConfigPreflightError, CLIError
from .inspect_bundle import render_json as _render_json


def _cli_error(
    error: TrainingRunConfigError,
    *,
    requested_path: str | PathLike[str],
    error_type: type[CLIError] = CLIError,
) -> CLIError:
    return error_type(
        error.reason_code,
        error.message,
        stage=error.stage,
        path=error.config_path or requested_path,
        source_path=error.source_path,
        frame_index=error.frame_index,
        sample_id=error.sample_id,
        template_id=error.template_id,
        config_field=error.field,
        split=error.split,
        underlying_reason_code=(
            error.original_reason_code or error.reason_code
        ),
        original_error=error,
    )


def validate_train_config(
    path: str | PathLike[str],
    *,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | PathLike[str] | None = None,
) -> ResolvedTrainingRun | ScratchTrainingPreparation:
    """Run the shared bundle or scratch preflight without creating a runtime."""

    try:
        config = load_effective_training_run_config(
            path, overrides, cli_cwd=cli_cwd
        )
    except TrainingRunConfigError as error:
        raise _cli_error(
            error,
            requested_path=path,
            error_type=CLIConfigPreflightError,
        ) from error
    if isinstance(config.model_source, ScratchModelSourceConfig):
        try:
            return prepare_scratch_training_run(config)
        except TrainingRunConfigError as error:
            raise _cli_error(
                error,
                requested_path=path,
                error_type=CLIConfigPreflightError,
            ) from error
    try:
        return resolve_training_run(config)
    except TrainingRunConfigError as error:
        raise _cli_error(error, requested_path=path) from error


def render_train_config_json(
    resolved: ResolvedTrainingRun | ResolvedScratchTrainingRun | ScratchTrainingPreparation,
) -> str:
    """Render strict deterministic JSON preflight metadata."""

    if not isinstance(
        resolved,
        (ResolvedTrainingRun, ResolvedScratchTrainingRun, ScratchTrainingPreparation),
    ):
        raise TypeError("resolved must be resolved training-run metadata")
    return _render_json(resolved.to_dict())


def _display(value: Any) -> str:
    if value is None:
        return "none"
    if type(value) is bool:
        return "yes" if value else "no"
    if type(value) in (str, int, float):
        return str(value)
    return _render_json(value)


def _label_lines(
    lines: list[str],
    *,
    split: str,
    statistics: Mapping[str, Mapping[str, int]],
) -> None:
    for term in ("energy", "forces", "stress"):
        values = statistics[term]
        lines.append(
            f"  {split}.{term}: present={values['present_frames']}, "
            f"missing={values['missing_frames']}, valid={values['valid_count']}"
        )


def render_train_config_human(
    resolved: ResolvedTrainingRun | ResolvedScratchTrainingRun | ScratchTrainingPreparation,
) -> str:
    """Render a deterministic, concise human-readable preflight report."""

    if not isinstance(
        resolved,
        (ResolvedTrainingRun, ResolvedScratchTrainingRun, ScratchTrainingPreparation),
    ):
        raise TypeError("resolved must be resolved training-run metadata")
    report = resolved.to_dict()
    if isinstance(resolved, ScratchTrainingPreparation):
        train = report["data"]["train"]
        validation = report["data"]["validation"]
        paths = report["runtime"]["paths"]
        lines = [
            "Reference-site MLIP scratch preflight",
            "Status: ready",
            f"Config schema: {report['schema_version']}",
            f"Config SHA-256: {report['config_fingerprint']}",
            f"Preparation SHA-256: {report['preparation_fingerprint']}",
            f"Registry SHA-256: {report['registry_fingerprint']}",
            f"Train semantic SHA-256: {train['semantic_digest']}",
            f"Validation semantic SHA-256: {validation['semantic_digest']}",
            "",
            "Data",
            f"  Train: {train['frame_count']} frames, {train['batch_count']} batches",
            "  Validation: "
            f"{validation['frame_count']} frames, {validation['batch_count']} batches",
            f"  Model species vocabulary: {_display(report['species_vocabulary'])}",
            "  Observed species vocabulary: "
            f"{_display(report['observed_species_vocabulary'])}",
            "",
            "Templates",
        ]
        for template_id in sorted(report["template_fingerprints"]):
            values = report["template_fingerprints"][template_id]
            lines.extend(
                [
                    f"  {template_id}: M={values['num_sites']}, "
                    f"phase={values['phase_approval_status']}, "
                    "evaluation_policy="
                    f"{'yes' if values['evaluation_policy_present'] else 'no'}",
                    "    Structural artifact: "
                    f"{values['structural_artifact_fingerprint']}",
                    f"    Full template: {values['full_template_fingerprint']}",
                    "    Phase specification: "
                    f"{values['phase_specification_fingerprint']}",
                    f"    Binding: {values['binding_fingerprint']}",
                ]
            )
        lines.extend(["", "Labels"])
        _label_lines(lines, split="train", statistics=train["label_statistics"])
        _label_lines(
            lines,
            split="validation",
            statistics=validation["label_statistics"],
        )
        lines.extend(
            [
                "",
                "Runtime",
                f"  Device: {report['runtime']['device']}",
                f"  Dtype: {report['runtime']['dtype']}",
                f"  Output directory: {paths['output_directory']}",
                "",
                "Full POSCAR/data/domain preflight completed.",
                "No model parameters, optimizer, initial bundle, output directory, "
                "or training run were created.",
                "Scratch training is available through refsite-mlip train.",
            ]
        )
        return "\n".join(lines)
    if isinstance(resolved, ResolvedScratchTrainingRun):
        source = report["model_source"]
        paths = report["runtime"]["paths"]
        template_ids = [
            item["builder"]["template_id"]
            for item in source["reference_templates"]
        ]
        return "\n".join(
            (
                "Reference-site MLIP scratch configuration",
                "Status: config ready",
                f"Config schema: {report['schema_version']}",
                f"Config SHA-256: {report['config_fingerprint']}",
                "Model source: scratch",
                f"Initialization seed: {source['initialization_seed']}",
                f"Default template: {source['default_template_id']}",
                f"Template IDs: {_display(template_ids)}",
                f"Device: {report['runtime']['device']}",
                f"Dtype: {report['runtime']['dtype']}",
                f"Output directory: {paths['output_directory']}",
                "Scratch execution requires full POSCAR/data preflight.",
                "No POSCAR/artifact/model construction or training was executed.",
            )
        )
    train = report["data"]["train"]
    validation = report["data"]["validation"]
    radii = report["radii"]
    user_radii = radii["user"]
    advanced = radii["advanced"]
    derived = radii["derived"]
    configuration = report["training_configuration"]
    paths = report["runtime"]["paths"]
    configured_paths = report["runtime"]["configured_paths"]
    template_ids = sorted(report["template_fingerprints"])
    lines = [
        "Reference-site MLIP training-run preflight",
        "Status: ready",
        f"Config schema: {report['schema_version']}",
        f"Config SHA-256: {report['config_fingerprint']}",
        f"Bundle SHA-256: {report['bundle_fingerprint']}",
        f"Train semantic SHA-256: {train['semantic_digest']}",
        f"Validation semantic SHA-256: {validation['semantic_digest']}",
        "",
        "Data",
        f"  Train: {train['frame_count']} frames, {train['batch_count']} batches",
        "  Validation: "
        f"{validation['frame_count']} frames, {validation['batch_count']} batches",
        f"  Species vocabulary: {_display(report['species_vocabulary'])}",
        f"  Train compositions: {_display(train['composition_statistics'])}",
        "  Validation compositions: "
        f"{_display(validation['composition_statistics'])}",
        "",
        "Templates",
    ]
    for template_id in template_ids:
        fingerprints = report["template_fingerprints"][template_id]
        lines.extend(
            [
                f"  {template_id}: "
                f"train_frames={train['template_frame_counts'].get(template_id, 0)}, "
                "validation_frames="
                f"{validation['template_frame_counts'].get(template_id, 0)}",
                "    Structural artifact: "
                f"{fingerprints['structural_artifact_fingerprint']}",
                f"    Full template: {fingerprints['full_template_fingerprint']}",
                "    Phase specification: "
                f"{fingerprints['phase_specification_fingerprint']}",
                f"    Binding: {fingerprints['binding_fingerprint']}",
                "    Evaluation policy: "
                f"{_display(fingerprints['evaluation_policy_fingerprint'])}",
            ]
        )
    lines.extend(["", "Labels"])
    _label_lines(lines, split="train", statistics=train["label_statistics"])
    _label_lines(
        lines,
        split="validation",
        statistics=validation["label_statistics"],
    )
    lines.extend(
        [
            "",
            "Interaction radii (angstrom)",
            f"  User: r_ot={user_radii['r_ot']}, r_mp={user_radii['r_mp']}",
            "  Advanced: "
            f"ot_switch_width={advanced['ot_switch_width']}, "
            f"ot_skin={advanced['ot_skin']}, mp_skin={advanced['mp_skin']}",
            "  Derived: "
            f"r_on_ot={derived['r_on_ot']}, r_off_ot={derived['r_off_ot']}, "
            f"r_candidate_ot={derived['r_candidate_ot']}, "
            f"r_candidate_mp={derived['r_candidate_mp']}",
            "",
            "Training configuration",
            f"  Loss: {_display(configuration['loss'])}",
            f"  Baseline: {_display(configuration['baseline'])}",
            f"  Optimizer: {_display(configuration['optimizer'])}",
            f"  Scheduler: {_display(configuration['scheduler'])}",
            f"  Selection: {_display(configuration['selection'])}",
            f"  Fit: {_display(configuration['fit'])}",
            "  Checkpointed fit: "
            f"{_display(configuration['checkpointed_fit'])}",
            "  Baseline preflight: "
            f"{_display(report['baseline_preflight'])}",
            "",
            "Runtime",
            f"  Device: {report['runtime']['device']}",
            f"  Dtype: {report['runtime']['dtype']}",
            f"  Seed: {report['runtime']['seed']}",
            "  Configured path expressions: "
            f"{_display(configured_paths)}",
            f"  Config path: {_display(paths['config'])}",
            f"  Initial bundle: {paths['initial_bundle']}",
            f"  Train inputs: {_display(paths['train_inputs'])}",
            f"  Validation inputs: {_display(paths['validation_inputs'])}",
            f"  Output directory: {paths['output_directory']}",
            f"  Expected checkpoints: {_display(report['expected_paths'])}",
            "",
            "No training was executed.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "render_train_config_human",
    "render_train_config_json",
    "validate_train_config",
]
