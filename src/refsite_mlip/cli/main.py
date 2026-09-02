"""Command-line entry point for refsite-mlip."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
import sys

from refsite_mlip import __version__

from .errors import CLIError, format_cli_error


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its process exit code."""

    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        return int(args.command_handler(args))
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
            original_error=error,
        )
        print(format_cli_error(wrapped), file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
