"""Small filesystem primitives shared by immutable binary artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AtomicCommitResult:
    """Outcome of a commit whose target is known to be durable by name."""

    temporary_cleanup_succeeded: bool
    orphaned_temporary: Path | None


def commit_temporary_file(
    temporary: Path,
    target: Path,
    *,
    overwrite: bool,
) -> AtomicCommitResult:
    """Commit a same-directory temporary file without a no-clobber race.

    ``os.link`` atomically creates the immutable target only when it does not
    exist.  A competing file or symlink therefore wins without ever being
    replaced.  Mutable targets retain the existing atomic ``os.replace``
    contract.
    """

    if overwrite:
        os.replace(temporary, target)
        return AtomicCommitResult(
            temporary_cleanup_succeeded=True,
            orphaned_temporary=None,
        )
    os.link(temporary, target)
    try:
        temporary.unlink()
    except OSError:
        # The link is the commit point.  A cleanup failure after it must not
        # turn a successful immutable save into a reported failure or remove
        # the target.  Callers may report/collect the orphan separately.
        return AtomicCommitResult(
            temporary_cleanup_succeeded=False,
            orphaned_temporary=temporary,
        )
    return AtomicCommitResult(
        temporary_cleanup_succeeded=True,
        orphaned_temporary=None,
    )


__all__ = ["AtomicCommitResult", "commit_temporary_file"]
