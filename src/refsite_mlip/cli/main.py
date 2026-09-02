"""Command-line entry point for refsite-mlip."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
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
