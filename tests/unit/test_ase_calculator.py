from __future__ import annotations

import pytest

from refsite_mlip.interfaces import (
    ASECalculatorConfig,
    ReferenceSiteASECalculatorError,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED


def test_ase_calculator_config_defaults_round_trip_and_validation():
    config = ASECalculatorConfig()
    assert config.solver_path == TRAIN_FIXED
    assert config.reuse_candidate_state
    assert config.collect_diagnostics
    assert config.sample_id == "ase:structure"
    assert config.origin_convention == "zero"
    assert ASECalculatorConfig.from_dict(config.to_dict()) == config

    adaptive = ASECalculatorConfig(
        solver_path=EVAL_ADAPTIVE,
        reuse_candidate_state=False,
        collect_diagnostics=False,
        sample_id="ase:test",
    )
    assert ASECalculatorConfig.from_dict(adaptive.to_dict()) == adaptive
    with pytest.raises(ValueError, match="solver_path"):
        ASECalculatorConfig(solver_path="invalid")
    with pytest.raises(TypeError, match="reuse_candidate_state"):
        ASECalculatorConfig(reuse_candidate_state=1)
    with pytest.raises(ValueError, match="sample_id"):
        ASECalculatorConfig(sample_id="")
    with pytest.raises(ValueError, match="origin_convention"):
        ASECalculatorConfig(origin_convention="cell")
    with pytest.raises(ValueError, match="unknown"):
        ASECalculatorConfig.from_dict({"unexpected": True})


def test_ase_calculator_error_retains_complete_context():
    cause = ValueError("underlying predictor failure")
    error = ReferenceSiteASECalculatorError(
        "UNSUPPORTED_SPECIES",
        "prediction failed",
        requested_properties=("energy", "forces"),
        template_id="alpha",
        solver_path=EVAL_ADAPTIVE,
        atom_count=3,
        species=(6, 41),
        composition=((6, 1), (41, 2)),
        predictor_reason_code="UNSUPPORTED_SPECIES",
        predictor_stage="structure_domain_preflight",
        original_error=cause,
    )
    assert error.reason_code == "UNSUPPORTED_SPECIES"
    assert error.requested_properties == ("energy", "forces")
    assert error.template_id == "alpha"
    assert error.solver_path == EVAL_ADAPTIVE
    assert error.atom_count == 3
    assert error.species == (6, 41)
    assert error.composition == ((6, 1), (41, 2))
    assert error.predictor_stage == "structure_domain_preflight"
    assert error.original_exception_type == "ValueError"
    assert "underlying predictor failure" in str(error)


def test_ase_calculator_config_is_frozen():
    config = ASECalculatorConfig()
    with pytest.raises(Exception):
        config.solver_path = EVAL_ADAPTIVE
