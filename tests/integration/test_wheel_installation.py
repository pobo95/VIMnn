from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterator, Mapping, Sequence

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUBPROCESS_TIMEOUT_SECONDS = 300


def _offline_environment(*, cuda_visible_devices: str = "") -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            # 10C-1 is an explicitly CPU-only gate.  Hiding any accelerator
            # also prevents a system-site PyTorch install from capturing or
            # consuming CUDA RNG state while the wheel workflow is exercised.
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _run_checked(
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [os.fspath(item) for item in argv],
        cwd=cwd,
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        command = " ".join(os.fspath(item) for item in argv)
        pytest.fail(
            f"installed-wheel command failed with exit {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


@dataclass(frozen=True)
class InstalledWheelEnvironment:
    """Repository-external environment containing only the built package wheel."""

    root: Path
    repository_root: Path
    wheel: Path
    python: Path
    console: Path
    environment: Mapping[str, str]

    def run(
        self,
        argv: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        working_directory = self.root / "work" if cwd is None else cwd
        result = subprocess.run(
            [os.fspath(item) for item in argv],
            cwd=working_directory,
            env=dict(self.environment),
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        if check and result.returncode != 0:
            command = " ".join(os.fspath(item) for item in argv)
            pytest.fail(
                f"installed command failed with exit {result.returncode}: {command}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def run_console(
        self,
        *arguments: str,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self.run((self.console, *arguments), cwd=cwd, check=check)

    def run_module(
        self,
        *arguments: str,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            (self.python, "-m", "refsite_mlip", *arguments),
            cwd=cwd,
            check=check,
        )


def build_installed_wheel_environment(
    root: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    cuda_visible_devices: str = "",
) -> InstalledWheelEnvironment:
    """Build offline, install non-editably, and return an isolated CLI runner."""

    root = root.resolve()
    repository_root = repository_root.resolve()
    if root == repository_root or repository_root in root.parents:
        raise ValueError("wheel test root must be outside the source repository")

    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    work = root / "work"
    work.mkdir()
    environment = _offline_environment(
        cuda_visible_devices=cuda_visible_devices
    )
    if cuda_visible_devices:
        # Surface asynchronous kernel failures in the command which launched
        # them.  This is a verification-only process setting and is not part
        # of any training or model configuration fingerprint.
        environment["CUDA_LAUNCH_BLOCKING"] = "1"

    _run_checked(
        (
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            wheelhouse,
            repository_root,
        ),
        cwd=work,
        environment=environment,
    )
    wheels = tuple(wheelhouse.glob("refsite_mlip-*.whl"))
    if len(wheels) != 1:
        pytest.fail(f"expected one refsite-mlip wheel, found {wheels!r}")

    virtual_environment = root / "venv"
    _run_checked(
        (sys.executable, "-m", "venv", "--system-site-packages", virtual_environment),
        cwd=work,
        environment=environment,
    )
    python = virtual_environment / "bin" / "python"
    console = virtual_environment / "bin" / "refsite-mlip"
    _run_checked(
        (
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-installed",
            wheels[0],
        ),
        cwd=work,
        environment=environment,
    )
    if not console.is_file():
        pytest.fail(f"wheel installation did not create console entry: {console}")

    return InstalledWheelEnvironment(
        root=root,
        repository_root=repository_root,
        wheel=wheels[0],
        python=python,
        console=console,
        environment=environment,
    )


@contextmanager
def installed_wheel_environment_context(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    cuda_visible_devices: str = "",
) -> Iterator[InstalledWheelEnvironment]:
    with tempfile.TemporaryDirectory(prefix="refsite-mlip-wheel-") as temporary:
        yield build_installed_wheel_environment(
            Path(temporary),
            repository_root=repository_root,
            cuda_visible_devices=cuda_visible_devices,
        )


@pytest.fixture(scope="session")
def installed_wheel_environment() -> Iterator[InstalledWheelEnvironment]:
    with installed_wheel_environment_context() as installed:
        yield installed


def test_wheel_installation_is_offline_and_source_tree_independent(
    installed_wheel_environment: InstalledWheelEnvironment,
) -> None:
    installed = installed_wheel_environment
    probe = installed.run(
        (
            installed.python,
            "-c",
            """
import importlib.metadata
import json
from pathlib import Path
import sys
import sysconfig
import refsite_mlip

distribution = importlib.metadata.distribution("refsite-mlip")
direct_url_text = distribution.read_text("direct_url.json")
console_entries = sorted(
    entry.name
    for entry in distribution.entry_points
    if entry.group == "console_scripts"
)
print(json.dumps({
    "console_entries": console_entries,
    "cwd": str(Path.cwd().resolve()),
    "direct_url": None if direct_url_text is None else json.loads(direct_url_text),
    "module": str(Path(refsite_mlip.__file__).resolve()),
    "purelib": str(Path(sysconfig.get_path("purelib")).resolve()),
    "sys_path": [str(Path(item or ".").resolve()) for item in sys.path],
    "version": refsite_mlip.__version__,
}, sort_keys=True))
""",
        )
    )
    metadata = json.loads(probe.stdout)
    module_path = Path(metadata["module"])
    purelib = Path(metadata["purelib"])
    source_package = installed.repository_root / "src" / "refsite_mlip"

    assert module_path.is_relative_to(purelib)
    assert not module_path.is_relative_to(source_package)
    assert metadata["cwd"] == str((installed.root / "work").resolve())
    assert str(installed.repository_root.resolve()) not in metadata["sys_path"]
    assert str((installed.repository_root / "src").resolve()) not in metadata["sys_path"]
    assert "refsite-mlip" in metadata["console_entries"]
    direct_url = metadata["direct_url"] or {}
    assert not direct_url.get("dir_info", {}).get("editable", False)

    console_version = installed.run_console("version")
    module_version = installed.run_module("version")
    expected = f"refsite-mlip {metadata['version']}\n"
    assert console_version.stdout == expected
    assert module_version.stdout == expected
    assert console_version.stderr == module_version.stderr
    assert "Traceback" not in console_version.stderr

    console_help = installed.run_console("--help")
    module_help = installed.run_module("--help")
    assert console_help.stdout == module_help.stdout
    assert console_help.stderr == module_help.stderr
    assert "Traceback" not in console_help.stderr
    assert "validate-train-config" in console_help.stdout
    assert "export-bundle" in console_help.stdout
