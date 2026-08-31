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
from .edge_list import (
    CompactTransportEdges,
    build_compact_transport_edges,
    materialize_dense_plan,
)
from .result import (
    DensePlanMaterialization,
    DualVariables,
    EvalOTConfig,
    OTResult,
    SparseOTResult,
    TrainSinkhornConfig,
)
from .sparse_sinkhorn import (
    solve_sparse_sinkhorn_train_fixed,
    sparse_marginal_residuals,
    sparse_sinkhorn_full_update,
    sparse_transport_plan,
)
from .support import (
    TRANSPORT_SUPPORT_CONVENTION_VERSION,
    TransportSupportConfig,
    TransportSupportDiagnostics,
    TransportSupportError,
    compact_c2_switch,
)

__all__ = [
    "DualVariables",
    "DensePlanMaterialization",
    "CompactTransportEdges",
    "MICImageDiagnostics",
    "EVAL_ADAPTIVE",
    "EvalOTConfig",
    "OTResult",
    "SparseOTResult",
    "TRAIN_FIXED",
    "TrainSinkhornConfig",
    "TRANSPORT_SUPPORT_CONVENTION_VERSION",
    "TransportSupportConfig",
    "TransportSupportDiagnostics",
    "TransportSupportError",
    "atom_site_cost",
    "atom_site_displacements",
    "build_compact_transport_edges",
    "compact_c2_switch",
    "minimum_image_diagnostics",
    "minimum_image_displacement",
    "materialize_dense_plan",
    "solve_atom_vacancy_ot",
    "solve_ragged_atom_vacancy_ot",
    "solve_sparse_sinkhorn_train_fixed",
    "sparse_marginal_residuals",
    "sparse_sinkhorn_full_update",
    "sparse_transport_plan",
]
