"""Command-line entry point for refsite-mlip."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
import sys
import traceback

from refsite_mlip import __version__

from .errors import (
    CLIConfigPreflightError,
    CLIError,
    CLIInterruptedError,
    format_cli_error,
)


def _add_debug_argument(parser: argparse.ArgumentParser, *, hidden: bool = False) -> None:
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS if hidden else False,
        help=(argparse.SUPPRESS if hidden else "show a traceback for runtime errors"),
    )


def _properties_argument(value: str) -> tuple[str, ...]:
    from .predict import normalize_properties

    try:
        return normalize_properties(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _terms_argument(value: str) -> tuple[str, ...]:
    from .evaluate import normalize_terms

    try:
        return normalize_terms(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _finite_float(value: str, *, positive: bool) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite number") from error
    invalid = result <= 0.0 if positive else result < 0.0
    if not math.isfinite(result) or invalid:
        qualifier = "positive" if positive else "nonnegative"
        raise argparse.ArgumentTypeError(f"must be finite and {qualifier}")
    return result


def _positive_float(value: str) -> float:
    return _finite_float(value, positive=True)


def _nonnegative_float(value: str) -> float:
    return _finite_float(value, positive=False)


def _device_argument(value: str) -> str:
    import re

    if re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", value) is None:
        raise argparse.ArgumentTypeError("must be cpu, cuda, or cuda:N")
    return value


def _add_training_config_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "config_path",
        nargs="?",
        help="training-run JSON or YAML configuration path",
    )
    source.add_argument(
        "--config",
        dest="config_option",
        metavar="PATH",
        help="alias for the positional configuration path",
    )
    parser.add_argument("--device", type=_device_argument, default=None)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default=None
    )
    parser.add_argument(
        "--max-epochs", type=_positive_integer, default=None, metavar="INTEGER"
    )
    parser.add_argument(
        "--batch-size", type=_positive_integer, default=None, metavar="INTEGER"
    )
    parser.add_argument(
        "--validation-batch-size",
        type=_positive_integer,
        default=None,
        metavar="INTEGER",
    )
    parser.add_argument(
        "--learning-rate", type=_positive_float, default=None, metavar="FLOAT"
    )
    parser.add_argument("--r-ot", type=_positive_float, default=None, metavar="FLOAT")
    parser.add_argument("--r-mp", type=_positive_float, default=None, metavar="FLOAT")
    parser.add_argument(
        "--output-directory", default=None, metavar="PATH"
    )


def _training_config_path(args: argparse.Namespace) -> str:
    return args.config_path if args.config_path is not None else args.config_option


def _training_config_overrides(args: argparse.Namespace):
    from refsite_mlip.config import (
        TrainingRunConfigError,
        TrainingRunConfigOverrides,
    )

    try:
        return TrainingRunConfigOverrides(
            device=args.device,
            dtype=args.dtype,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            validation_batch_size=args.validation_batch_size,
            learning_rate=args.learning_rate,
            r_ot=args.r_ot,
            r_mp=args.r_mp,
            output_directory=args.output_directory,
        )
    except TrainingRunConfigError as error:
        raise CLIConfigPreflightError(
            error.reason_code,
            error.message,
            stage=error.stage,
            path=_training_config_path(args),
            config_field=error.field,
            underlying_reason_code=error.reason_code,
            original_error=error,
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refsite-mlip",
        description="Reference-site MLIP portable-model utilities.",
    )
    _add_debug_argument(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    version = commands.add_parser("version", help="print the package version")
    _add_debug_argument(version, hidden=True)
    version.set_defaults(command_handler=_run_version)

    inspect = commands.add_parser(
        "inspect-bundle",
        help="safely validate and inspect a portable model bundle",
        description=(
            "Load a portable bundle with the weights-only safe loader and print "
            "validated public metadata without model inference."
        ),
    )
    inspect.add_argument("bundle_path", help="portable model bundle path")
    inspect.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit deterministic compact JSON",
    )
    _add_debug_argument(inspect, hidden=True)
    inspect.set_defaults(command_handler=_run_inspect_bundle)

    predict = commands.add_parser(
        "predict",
        help="predict energy/forces/stress for extxyz frames",
        description=(
            "Load one portable bundle runtime, predict ordered extxyz frames, "
            "and atomically write ASE SinglePointCalculator results."
        ),
    )
    predict.add_argument("--bundle", required=True, dest="bundle_path")
    predict.add_argument("--input", required=True, dest="input_path")
    predict.add_argument("--output", required=True, dest="output_path")
    predict.add_argument(
        "--index",
        default=":",
        help="ASE extxyz index expression (default: :)",
    )
    templates = predict.add_mutually_exclusive_group()
    templates.add_argument(
        "--template-id",
        help="one exact bundle template ID for every selected frame",
    )
    templates.add_argument(
        "--template-key",
        help="Atoms.info key containing each frame's exact template ID",
    )
    predict.add_argument(
        "--solver",
        choices=("train-fixed", "eval-adaptive"),
        default="train-fixed",
    )
    predict.add_argument(
        "--properties",
        type=_properties_argument,
        default=("energy", "forces"),
        metavar="LIST",
        help="comma-separated energy,forces,stress (default: energy,forces)",
    )
    predict.add_argument(
        "--device",
        type=_device_argument,
        default="cpu",
        metavar="DEVICE",
    )
    predict.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    predict.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=8,
        metavar="INTEGER",
    )
    predict.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing regular output file",
    )
    predict.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit deterministic compact JSON summary",
    )
    _add_debug_argument(predict, hidden=True)
    predict.set_defaults(command_handler=_run_predict)

    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate labeled extxyz frames without training",
        description=(
            "Run one portable bundle predictor over labeled extxyz frames and "
            "report masked physical metrics and normalized loss."
        ),
    )
    evaluate.add_argument("--bundle", required=True, dest="bundle_path")
    evaluate.add_argument("--input", required=True, dest="input_path")
    evaluate.add_argument(
        "--index", default=":", help="ASE extxyz index expression (default: :)"
    )
    evaluation_templates = evaluate.add_mutually_exclusive_group()
    evaluation_templates.add_argument(
        "--template-id",
        help="one exact bundle template ID for every selected frame",
    )
    evaluation_templates.add_argument(
        "--template-key",
        help="Atoms.info key containing each frame's exact template ID",
    )
    evaluate.add_argument(
        "--solver",
        choices=("train-fixed", "eval-adaptive"),
        default="train-fixed",
    )
    evaluate.add_argument(
        "--terms",
        type=_terms_argument,
        default=("energy", "forces"),
        metavar="LIST",
        help="comma-separated energy,forces,stress (default: energy,forces)",
    )
    evaluate.add_argument(
        "--device", type=_device_argument, default="cpu", metavar="DEVICE"
    )
    evaluate.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    evaluate.add_argument(
        "--batch-size", type=_positive_integer, default=8, metavar="INTEGER"
    )
    evaluate.add_argument(
        "--energy-mode",
        choices=("per-structure", "per-atom"),
        default="per-structure",
    )
    for term in ("energy", "force", "stress"):
        evaluate.add_argument(
            f"--{term}-scale",
            type=_positive_float,
            default=1.0,
            metavar="FLOAT",
        )
        evaluate.add_argument(
            f"--{term}-weight",
            type=_nonnegative_float,
            default=1.0,
            metavar="FLOAT",
        )
    evaluate.add_argument(
        "--output",
        dest="output_path",
        help="atomically write the deterministic JSON report to PATH",
    )
    evaluate.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing regular report file",
    )
    evaluate.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit deterministic compact JSON when writing to stdout",
    )
    _add_debug_argument(evaluate, hidden=True)
    evaluate.set_defaults(command_handler=_run_evaluate)

    validate_train = commands.add_parser(
        "validate-train-config",
        help="validate a canonical training-run config without training",
        description=(
            "Safely verify a portable initial bundle, extxyz data, radii, and "
            "training controls without model execution or filesystem writes."
        ),
    )
    _add_training_config_arguments(validate_train)
    validate_train.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit deterministic compact JSON preflight metadata",
    )
    _add_debug_argument(validate_train, hidden=True)
    validate_train.set_defaults(command_handler=_run_validate_train_config)

    train = commands.add_parser(
        "train",
        help="execute a validated fresh training run",
        description=(
            "Preflight a canonical training-run config and compose the existing "
            "baseline, optimizer, scheduler, and checkpointed-fit engine."
        ),
    )
    _add_training_config_arguments(train)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="stop after the same read-only preflight as validate-train-config",
    )
    train.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit only deterministic final JSON on stdout",
    )
    train.add_argument(
        "--quiet",
        action="store_true",
        help="suppress presentation-only training progress on stderr",
    )
    _add_debug_argument(train, hidden=True)
    train.set_defaults(command_handler=_run_train)

    resume = commands.add_parser(
        "resume",
        help="continue an existing run from its managed latest checkpoint",
        description=(
            "Validate an existing training run directory, restore only its "
            "weights-only-safe checkpoints/latest.pt, and continue the shared "
            "checkpointed-fit engine with a strictly increased max epoch."
        ),
    )
    resume.add_argument(
        "run_directory", help="run directory created by refsite-mlip train"
    )
    resume.add_argument(
        "--max-epochs",
        required=True,
        type=_positive_integer,
        metavar="INTEGER",
        help="strictly increased terminal epoch count",
    )
    resume.add_argument(
        "--dry-run",
        action="store_true",
        help="perform complete read-only resume preflight without acquiring a lock",
    )
    resume.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit only deterministic final JSON on stdout",
    )
    resume.add_argument(
        "--quiet",
        action="store_true",
        help="suppress presentation-only training progress on stderr",
    )
    _add_debug_argument(resume, hidden=True)
    resume.set_defaults(command_handler=_run_resume)

    export = commands.add_parser(
        "export-bundle",
        help="export a portable model bundle from a managed training checkpoint",
        description=(
            "Safely validate a train/resume run and strip optimizer, scheduler, "
            "history, data, and RNG state from its managed best or latest "
            "checkpoint without model execution."
        ),
    )
    export.add_argument(
        "run_directory", help="run directory created by refsite-mlip train"
    )
    export.add_argument(
        "--source",
        required=True,
        choices=("best", "latest"),
        help="managed checkpoint alias to export",
    )
    export.add_argument(
        "--output",
        required=True,
        dest="output_path",
        help="portable model bundle output path",
    )
    export.add_argument(
        "--initial-bundle",
        dest="initial_bundle_path",
        help=(
            "replacement initial bundle path; its semantic SHA-256 must exactly "
            "match the stored run metadata"
        ),
    )
    export.add_argument(
        "--dry-run",
        action="store_true",
        help="validate through strict model-state load and capture without saving",
    )
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing regular output bundle",
    )
    export.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit deterministic compact JSON",
    )
    _add_debug_argument(export, hidden=True)
    export.set_defaults(command_handler=_run_export_bundle)
    return parser


def _run_version(args: argparse.Namespace) -> int:
    del args
    print(f"refsite-mlip {__version__}")
    return 0


def _run_inspect_bundle(args: argparse.Namespace) -> int:
    # Keep version/help lightweight: importing torch and bundle machinery is
    # deferred until this command is actually selected.
    from .inspect_bundle import inspect_bundle, render_human, render_json

    report = inspect_bundle(args.bundle_path)
    output = render_json(report) if args.json_output else render_human(report)
    print(output)
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    from .predict import (
        ExtXYZPredictionConfig,
        predict_extxyz,
        render_prediction_human,
        render_prediction_json,
    )

    config = ExtXYZPredictionConfig(
        bundle_path=args.bundle_path,
        input_path=args.input_path,
        output_path=args.output_path,
        index=args.index,
        template_id=args.template_id,
        template_key=args.template_key,
        solver_path=args.solver,
        properties=args.properties,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    report = predict_extxyz(config)
    output = (
        render_prediction_json(report)
        if args.json_output
        else render_prediction_human(report)
    )
    print(output)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import (
        ExtXYZEvaluationConfig,
        evaluate_extxyz,
        render_evaluation_human,
        render_evaluation_json,
    )

    config = ExtXYZEvaluationConfig(
        bundle_path=args.bundle_path,
        input_path=args.input_path,
        index=args.index,
        template_id=args.template_id,
        template_key=args.template_key,
        solver_path=args.solver,
        terms=args.terms,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        energy_mode=args.energy_mode,
        energy_scale=args.energy_scale,
        force_scale=args.force_scale,
        stress_scale=args.stress_scale,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        stress_weight=args.stress_weight,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )
    report = evaluate_extxyz(config)
    if args.output_path is None:
        output = (
            render_evaluation_json(report)
            if args.json_output
            else render_evaluation_human(report)
        )
        print(output)
    return 0


def _run_validate_train_config(args: argparse.Namespace) -> int:
    from .validate_train_config import (
        render_train_config_human,
        render_train_config_json,
        validate_train_config,
    )

    resolved = validate_train_config(
        _training_config_path(args),
        overrides=_training_config_overrides(args),
    )
    output = (
        render_train_config_json(resolved)
        if args.json_output
        else render_train_config_human(resolved)
    )
    print(output)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from .train import (
        render_train_result_human,
        render_train_result_json,
        run_training,
    )
    from .training_progress import TrainingProgressConfig, TrainingProgressRenderer

    progress_renderer = TrainingProgressRenderer(
        TrainingProgressConfig(enabled=not args.quiet),
        stream=sys.stderr,
    )
    try:
        result = run_training(
            _training_config_path(args),
            dry_run=args.dry_run,
            progress_renderer=progress_renderer,
            overrides=_training_config_overrides(args),
        )
        output = (
            render_train_result_json(result)
            if args.json_output
            else render_train_result_human(result)
        )
        print(output)
        return 0
    finally:
        progress_renderer.close_log()


def _run_resume(args: argparse.Namespace) -> int:
    from .resume import (
        render_resume_human,
        render_resume_json,
        resume_training,
    )
    from .training_progress import TrainingProgressConfig, TrainingProgressRenderer

    progress_renderer = TrainingProgressRenderer(
        TrainingProgressConfig(enabled=not args.quiet),
        stream=sys.stderr,
    )
    try:
        result = resume_training(
            args.run_directory,
            max_epochs=args.max_epochs,
            dry_run=args.dry_run,
            progress_renderer=progress_renderer,
        )
        output = (
            render_resume_json(result)
            if args.json_output
            else render_resume_human(result)
        )
        print(output)
        return 0
    finally:
        progress_renderer.close_log()


def _run_export_bundle(args: argparse.Namespace) -> int:
    from .export_bundle import (
        ExportBundleConfig,
        export_bundle,
        render_export_bundle_human,
        render_export_bundle_json,
    )

    result = export_bundle(
        ExportBundleConfig(
            run_directory=args.run_directory,
            source=args.source,
            output_path=args.output_path,
            initial_bundle_path=args.initial_bundle_path,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    )
    output = (
        render_export_bundle_json(result)
        if args.json_output
        else render_export_bundle_human(result)
    )
    print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its process exit code."""

    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        return int(args.command_handler(args))
    except CLIInterruptedError as error:
        if args.debug:
            traceback.print_exception(error, file=sys.stderr)
        else:
            print(format_cli_error(error), file=sys.stderr)
        return 130
    except CLIConfigPreflightError as error:
        if args.debug:
            traceback.print_exception(error, file=sys.stderr)
        else:
            print(format_cli_error(error), file=sys.stderr)
        return 2
    except CLIError as error:
        if args.debug:
            raise
        print(format_cli_error(error), file=sys.stderr)
        return 1
    except Exception as error:
        if args.debug:
            raise
        wrapped = CLIError(
            "CLI_RUNTIME_ERROR",
            "command failed unexpectedly",
            stage=f"command.{args.command}",
            bundle_path=getattr(args, "bundle_path", None),
            path=(
                getattr(args, "config_path", None)
                or getattr(args, "config_option", None)
                or getattr(args, "run_directory", None)
            ),
            original_error=error,
        )
        print(format_cli_error(wrapped), file=sys.stderr)
        return 1
    except KeyboardInterrupt as error:
        if args.debug:
            traceback.print_exception(error, file=sys.stderr)
        else:
            wrapped = CLIInterruptedError(
                "COMMAND_INTERRUPTED",
                "command was interrupted",
                stage=f"command.{args.command}",
                path=(
                    getattr(args, "config_path", None)
                    or getattr(args, "config_option", None)
                    or getattr(args, "run_directory", None)
                ),
                original_error=error,
            )
            print(format_cli_error(wrapped), file=sys.stderr)
        return 130


__all__ = ["build_parser", "main"]
