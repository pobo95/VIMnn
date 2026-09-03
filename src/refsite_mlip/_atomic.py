"""Small filesystem primitives shared by immutable binary artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def commit_temporary_file(
    temporary: Path,
    target: Path,
    *,
    overwrite: bool,
) -> None:
    """Commit a same-directory temporary file without a no-clobber race.

    ``os.link`` atomically creates the immutable target only when it does not
    exist.  A competing file or symlink therefore wins without ever being
    replaced.  Mutable targets retain the existing atomic ``os.replace``
    contract.
    """

    if overwrite:
        os.replace(temporary, target)
        return
    os.link(temporary, target)
    temporary.unlink()


__all__ = ["commit_temporary_file"]
