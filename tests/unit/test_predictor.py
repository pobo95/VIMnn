from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from refsite_mlip.inference import (
    BatchPrediction,
    PredictorConfig,
    StructurePrediction,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED


def _prediction() -> BatchPrediction:
    energy = torch.tensor([1.0, 2.0], dtype=torch.float64)
    return BatchPrediction(
        energy=energy,
        baseline_energy=torch.tensor([0.25, 0.5], dtype=torch.float64),
        residual_energy=torch.tensor([0.75, 1.5], dtype=torch.float64),
        forces=torch.arange(15, dtype=torch.float64).reshape(5, 3),
        stress=torch.arange(18, dtype=torch.float64).reshape(2, 3, 3),
        stress_voigt=torch.arange(12, dtype=torch.float64).reshape(2, 6),
        site_energy=torch.arange(7, dtype=torch.float64),
        atom_ptr=torch.tensor([0, 2, 5], dtype=torch.long),
        site_ptr=torch.tensor([0, 3, 7], dtype=torch.long),
        sample_ids=("first", "second"),
        template_ids=("alpha", "zeta"),
        diagnostics=({"selected": 1}, None),
    )


def test_predictor_config_validation_and_round_trip():
    config = PredictorConfig(
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
        return_aux=True,
        return_candidate_neighbor_states=True,
        output_device="cpu",
    )
    assert PredictorConfig.from_dict(config.to_dict()) == config
    assert PredictorConfig().solver_path == TRAIN_FIXED
    for values in (
        {"solver_path": "unknown"},
        {"output_device": "cuda"},
    ):
        with pytest.raises(ValueError):
            PredictorConfig(**values)
    with pytest.raises(TypeError):
        PredictorConfig(compute_forces=1)
    with pytest.raises(ValueError):
        PredictorConfig.from_dict({"extra": True})


def test_batch_prediction_structure_views_and_dictionary_access():
    prediction = _prediction()
    assert len(prediction) == 2
    assert prediction["energy"] is prediction.energy
    first = prediction[0]
    second = prediction.structure(-1)
    assert isinstance(first, StructurePrediction)
    assert first.sample_id == "first" and first.template_id == "alpha"
    assert first.energy.shape == ()
    assert first.forces.shape == (2, 3)
    assert first.site_energy.shape == (3,)
    assert second.sample_id == "second"
    assert second.forces.shape == (3, 3)
    assert second.site_energy.shape == (4,)
    structures = prediction.structures
    assert tuple(value.sample_id for value in structures) == ("first", "second")
    assert torch.equal(structures[0].forces, first.forces)
    assert torch.equal(structures[1].forces, second.forces)
    assert prediction.auxiliary is prediction.diagnostics
    with pytest.raises(IndexError):
        prediction.structure(2)


def test_batch_prediction_rejects_graph_nonfinite_and_bad_ragged_contract():
    prediction = _prediction()
    requiring_grad = prediction.energy.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="detached"):
        replace(prediction, energy=requiring_grad)
    with pytest.raises(ValueError, match="NaN or Inf"):
        replace(prediction, energy=torch.tensor([float("nan"), 2.0]))
    with pytest.raises(ValueError, match="atom_ptr"):
        replace(prediction, atom_ptr=torch.tensor([0, 5], dtype=torch.long))
    with pytest.raises(ValueError, match="original sample order"):
        replace(
            prediction,
            candidate_reuse_decisions={"second": object(), "first": object()},
        )
