"""Narrow compatibility helpers for the pinned PyTorch/e3nn environment."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Tuple

import torch


def import_e3nn_0_4_4() -> Tuple[ModuleType, ModuleType]:
    """Import trusted e3nn 0.4.4 constants under a scoped safe-global context.

    PyTorch 2.6 changed torch.load to weights_only=True by default. The trusted
    e3nn 0.4.4 package constants contain the built-in slice type. The context
    adds only slice while e3nn imports and restores every caller-owned safe
    global afterwards. It never disables weights_only or modifies site-packages.
    """

    if "e3nn.o3" in sys.modules:
        import e3nn
        from e3nn import o3

        return e3nn, o3

    caller_globals = list(torch.serialization.get_safe_globals())
    if slice in caller_globals:
        import e3nn
        from e3nn import o3
    else:
        with torch.serialization.safe_globals([slice]):
            import e3nn
            from e3nn import o3

    return e3nn, o3
