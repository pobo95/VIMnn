"""Dense balanced aggregate-vacancy entropic transport."""

from .batch import solve_ragged_atom_vacancy_ot
from .cost import (
    MICImageDiagnostics,
    atom_site_cost,
    atom_site_displacements,
    minimum_image_diagnostics,
    minimum_image_displacement,
)
from .factory import EVAL_ADAPTIVE, TRAIN_FIXED, solve_atom_vacancy_ot
from .result import DualVariables, EvalOTConfig, OTResult, TrainSinkhornConfig
from .support import (
    TRANSPORT_SUPPORT_CONVENTION_VERSION,
    TransportSupportConfig,
    TransportSupportDiagnostics,
    TransportSupportError,
    compact_c2_switch,
)

__all__ = [
    "DualVariables",
    "MICImageDiagnostics",
    "EVAL_ADAPTIVE",
    "EvalOTConfig",
    "OTResult",
    "TRAIN_FIXED",
    "TrainSinkhornConfig",
    "TRANSPORT_SUPPORT_CONVENTION_VERSION",
    "TransportSupportConfig",
    "TransportSupportDiagnostics",
    "TransportSupportError",
    "atom_site_cost",
    "atom_site_displacements",
    "compact_c2_switch",
    "minimum_image_diagnostics",
    "minimum_image_displacement",
    "solve_atom_vacancy_ot",
    "solve_ragged_atom_vacancy_ot",
]
