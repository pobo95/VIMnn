from __future__ import annotations

import pytest
import torch

from refsite_mlip.transport import TRAIN_FIXED, TrainSinkhornConfig, solve_atom_vacancy_ot
from refsite_mlip.transport.operating_point import OTOperatingDomain, audit_train_fixed_operating_point


def _domain():
    return OTOperatingDomain(0.5, 1.5, "float64", 256, 1.0e-7)


def test_supported_synthetic_operating_point_and_metadata():
    domain = _domain()
    cost = torch.tensor([[0.01, 0.9, 1.3], [0.8, 0.02, 0.7], [1.1, 0.6, 0.03], [0.4, 0.5, 0.8]], dtype=torch.float64)
    result = solve_atom_vacancy_ot(cost, domain.epsilon_ot, TRAIN_FIXED, "sinkhorn", TrainSinkhornConfig(256))
    audit = audit_train_fixed_operating_point(result, cost, domain, structure_id="supported")
    assert audit.residual <= 1.0e-7
    assert audit.vacancy_mass_error <= 1.0e-7
    assert domain.to_dict()["solver_contract_version"] == "dense_aggregate_vacancy_ot_v1"


def test_bad_fixed_operating_point_is_not_repaired():
    domain = _domain()
    cost = torch.tensor([[0.0, 80.0], [80.0, 0.0], [40.0, 40.0]], dtype=torch.float64)
    result = solve_atom_vacancy_ot(cost, 0.02, TRAIN_FIXED, "sinkhorn", TrainSinkhornConfig(2))
    with pytest.raises(ValueError, match="residual|iteration metadata"):
        audit_train_fixed_operating_point(result, cost, domain, structure_id="unsupported_extreme")
