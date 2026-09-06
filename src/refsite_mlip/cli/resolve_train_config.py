"""CLI support for compiling beginner recipes into canonical schema v2."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from typing import Any

from refsite_mlip._atomic import commit_temporary_file
from refsite_mlip.config import (
    ResolvedTrainingRecipe,
    TrainingRecipeError,
    TrainingRunConfigOverrides,
    resolve_training_recipe,
)

from .errors import CLIConfigPreflightError
from .inspect_bundle import render_json


def _cli_error(error: TrainingRecipeError, path: str | os.PathLike[str]) -> CLIConfigPreflightError:
    return CLIConfigPreflightError(
        error.reason_code,
        error.message,
        stage=error.stage,
        path=error.path or path,
        config_field=error.field,
        underlying_reason_code=error.reason_code,
        original_error=error,
    )


def _same_file_or_path(first: Path, second: Path) -> bool:
    try:
        if first.resolve(strict=False) == second.resolve(strict=False):
            return True
        if first.exists() and second.exists() and os.path.samefile(first, second):
            return True
    except OSError:
        pass
    return False


def _atomic_text(target: Path, encoded: str, *, overwrite: bool) -> None:
    if target.is_symlink():
        raise CLIConfigPreflightError(
            "OUTPUT_SYMLINK_REJECTED", "resolved output must not be a symlink",
            stage="recipe.output", path=target
        )
    if target.exists() and (not target.is_file() or not overwrite):
        raise CLIConfigPreflightError(
            "OUTPUT_EXISTS", "resolved output already exists and was not replaced",
            stage="recipe.output", path=target
        )
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
    except OSError as error:
        raise CLIConfigPreflightError(
            "OUTPUT_TEMPFILE_FAILED", "could not create same-directory temporary file",
            stage="recipe.output", path=target, original_error=error
        ) from error
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if target.is_symlink():
            raise CLIConfigPreflightError(
                "OUTPUT_SYMLINK_REJECTED", "resolved output became a symlink before commit",
                stage="recipe.output", path=target
            )
        try:
            commit_temporary_file(temporary, target, overwrite=overwrite)
        except FileExistsError as error:
            raise CLIConfigPreflightError(
                "OUTPUT_EXISTS", "a competing writer created the output; it was preserved",
                stage="recipe.output", path=target, original_error=error
            ) from error
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def resolve_train_config(
    recipe_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    overrides: TrainingRunConfigOverrides | None = None,
    cli_cwd: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> ResolvedTrainingRecipe:
    """Compile and optionally save the canonical config plus resolution manifest."""

    if type(dry_run) is not bool or type(overwrite) is not bool:
        raise TypeError("dry_run and overwrite must be bool")
    if not dry_run and (output_path is None or manifest_path is None):
        raise CLIConfigPreflightError(
            "MISSING_RESOLUTION_OUTPUT",
            "--output and --manifest are required unless --dry-run is used",
            stage="recipe.output",
            path=recipe_path,
        )

    try:
        effective_cli_cwd = Path.cwd() if cli_cwd is None else Path(cli_cwd)
        resolved = resolve_training_recipe(
            recipe_path, overrides=overrides, cli_cwd=effective_cli_cwd
        )
        if (
            overrides is not None
            and overrides.output_directory is not None
            and not Path(overrides.output_directory).is_absolute()
        ):
            recipe_parent = Path(str(resolved.recipe.source_path)).parent
            absolute_output = (
                effective_cli_cwd / overrides.output_directory
            ).resolve(strict=False)
            rebased_output = os.path.relpath(absolute_output, recipe_parent)
            canonical = replace(
                resolved.config,
                output_directory=rebased_output,
                output_directory_base=None,
            )
            resolved = ResolvedTrainingRecipe(
                canonical,
                replace(
                    resolved.manifest,
                    compiled_config_fingerprint=canonical.config_fingerprint,
                ),
                resolved.recipe,
                resolved.automatic_reference_preparation,
            )
    except TrainingRecipeError as error:
        raise _cli_error(error, recipe_path) from error
    if dry_run and output_path is None and manifest_path is None:
        return resolved
    if output_path is None or manifest_path is None:
        raise CLIConfigPreflightError(
            "MISSING_RESOLUTION_OUTPUT",
            "--output and --manifest must be provided together",
            stage="recipe.output",
            path=recipe_path,
        )
    output = Path(output_path)
    manifest = Path(manifest_path)
    recipe = Path(recipe_path)
    recipe_parent = Path(str(resolved.recipe.source_path)).parent.resolve(
        strict=True
    )
    if output.parent.resolve(strict=False) != recipe_parent:
        raise CLIConfigPreflightError(
            "OUTPUT_CONFIG_BASE_MISMATCH",
            "canonical config output must share the recipe directory so relative path semantics remain unchanged",
            stage="recipe.output",
            path=output,
        )
    if _same_file_or_path(output, manifest):
        raise CLIConfigPreflightError(
            "OUTPUT_COLLISION", "resolved config and manifest must be different files",
            stage="recipe.output", path=output
        )
    protected = [recipe]
    base = Path(resolved.recipe.source_path).parent
    for source in resolved.recipe.reference.sources:
        if source.specification is not None:
            protected.append(base / source.specification)
        protected.append(base / source.poscar)
    for sources in (resolved.recipe.data.train, resolved.recipe.data.validation):
        protected.extend(base / source.path for source in sources)
    for target in (output, manifest):
        if any(_same_file_or_path(target, source) for source in protected):
            raise CLIConfigPreflightError(
                "INPUT_OUTPUT_COLLISION", "recipe output collides with a semantic input",
                stage="recipe.output", path=target
            )
        if not target.parent.exists() or not target.parent.is_dir():
            raise CLIConfigPreflightError(
                "OUTPUT_PARENT_NOT_FOUND", "output parent directory must already exist",
                stage="recipe.output", path=target.parent
            )
        if target.is_symlink():
            raise CLIConfigPreflightError(
                "OUTPUT_SYMLINK_REJECTED",
                "resolved output must not be a symlink",
                stage="recipe.output",
                path=target,
            )
        if target.exists() and (not target.is_file() or not overwrite):
            raise CLIConfigPreflightError(
                "OUTPUT_EXISTS",
                "resolved output already exists and was not replaced",
                stage="recipe.output",
                path=target,
            )
    if not dry_run:
        _atomic_text(output, resolved.config.canonical_json(), overwrite=overwrite)
        try:
            _atomic_text(
                manifest,
                render_json(resolved.manifest.to_dict()),
                overwrite=overwrite,
            )
        except Exception:
            # The config is an independently valid immutable artifact.  Never
            # roll it back after its commit point.
            raise
    return resolved


def render_resolution_json(resolved: ResolvedTrainingRecipe) -> str:
    return render_json(resolved.to_dict())


def render_resolution_human(resolved: ResolvedTrainingRecipe) -> str:
    config = resolved.config
    source = config.model_source
    templates = [] if source is None else [item.template_id for item in source.reference_templates]
    radii = config.radii.derived
    maximum_order = (
        resolved.recipe.model.correlation
        if resolved.recipe.model.correlation_method == "symmetric"
        else 3
    )
    lines = [
            "Reference-site MLIP training recipe resolution",
            "Status: resolved (no training executed)",
            f"Recipe SHA-256: {resolved.recipe.content_fingerprint}",
            f"Canonical configuration SHA-256: {config.config_fingerprint}",
            f"Manifest SHA-256: {resolved.manifest.content_fingerprint}",
            f"Correlation method: {resolved.recipe.model.correlation_method}",
            f"Maximum correlation order: {maximum_order}",
            f"Training OT solver: {resolved.manifest.training_ot_solver}",
            f"Inference OT solver: {resolved.manifest.inference_ot_solver}",
            "Application: prediction/evaluation call-time preference",
            f"Sinkhorn iterations: {resolved.manifest.sinkhorn_iterations}",
            "Residual tolerance: "
            f"{resolved.manifest.sinkhorn_residual_tolerance:.9g}",
            f"Nonconvergence policy: {resolved.manifest.nonconvergence_policy}",
            f"Training batch size: {config.data.batch_size}",
            "Validation batch size: "
            f"{config.data.effective_validation_batch_size}",
            f"Templates: {', '.join(templates)}",
            f"Radii: r_ot={config.radii.r_ot}, r_mp={config.radii.r_mp}, "
            f"r_candidate_ot={radii.r_candidate_ot}, r_candidate_mp={radii.r_candidate_mp}",
            f"Output directory: {config.output_directory}",
    ]
    automatic = resolved.automatic_reference_preparation
    if automatic is not None:
        lines.extend(("", f"References: {len(automatic.results)}"))
        for result in automatic.results:
            certificate = result.to_dict()
            strain = certificate["strain"]
            vacancies = certificate["vacancies"]
            train_ids = vacancies["train"]["sample_ids"]
            validation_ids = vacancies["validation"]["sample_ids"]
            observed_k = sorted(
                set(vacancies["train"]["observed_K_values"])
                | set(vacancies["validation"]["observed_K_values"])
            )
            lines.extend(
                (
                    f"  Template {result.template_id}",
                    f"    POSCAR: {result.original_poscar}",
                    f"    M: {certificate['num_reference_sites']}",
                    f"    Assigned train frames: {len(train_ids)}",
                    f"    Assigned validation frames: {len(validation_ids)}",
                    "    Observed maximum strain: "
                    f"{strain['observed_all']:.17g}",
                    "    Resolved maximum strain: "
                    f"{strain['resolved_maximum_strain']:.17g}",
                    f"    Vacancy range: K={observed_k}",
                    f"    Phase approval: {certificate['approval_status']}",
                    "    Specification SHA-256: "
                    f"{certificate['specification_sha256']}",
                    f"    Artifact SHA-256: {certificate['artifact_sha256']}",
                )
            )
            evaluation = certificate.get("evaluation_certificate")
            if evaluation is not None:
                split_counts = evaluation["audit_input"]["split_counts"]
                probes = evaluation["derivative_probes"]
                lines.extend(
                    (
                        "    Evaluation solver: sinkhorn_newton_krylov",
                        f"    Policy status: {evaluation['status']}",
                        f"    Scope: {evaluation['scope']}",
                        f"    Phase approval: {evaluation['phase_approval']}",
                        "    Audited train/validation frames: "
                        f"{split_counts['train']}/{split_counts['validation']}",
                        "    Position/strain probes: "
                        f"{sum(item['position_directions'] for item in probes)}/"
                        f"{sum(item['strain_directions'] for item in probes)}",
                        f"    Fallback count: {evaluation['fallback_count']}",
                        "    Policy SHA-256: "
                        f"{evaluation['policy_fingerprint']}",
                        "    Evaluation certificate SHA-256: "
                        f"{evaluation['evaluation_certificate_sha256']}",
                        "    Production/MD guarantee: no",
                    )
                )
        lines.extend(("", "No training was executed."))
    return "\n".join(lines)


__all__ = [
    "render_resolution_human",
    "render_resolution_json",
    "resolve_train_config",
]
