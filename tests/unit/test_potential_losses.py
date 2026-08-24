from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import LossConfig, compute_potential_loss


def _batch(
    counts,
    *,
    dtype=torch.float64,
    device="cpu",
    energy=None,
    energy_mask=None,
    forces=None,
    force_mask=None,
    force_present=None,
    stress=None,
    stress_mask=None,
    stress_present=None,
):
    device = torch.device(device)
    counts_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    atom_ptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(counts_tensor, dim=0),
        )
    )
    batch_size = len(counts)
    num_atoms = sum(counts)
    atom_batch = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=device), counts_tensor
    )

    def floating(value, shape):
        if value is None:
            return torch.zeros(shape, dtype=dtype, device=device)
        return torch.as_tensor(value, dtype=dtype, device=device)

    def boolean(value, shape):
        if value is None:
            return torch.zeros(shape, dtype=torch.bool, device=device)
        return torch.as_tensor(value, dtype=torch.bool, device=device)

    energy_value = floating(energy, (batch_size,))
    energy_valid = boolean(energy_mask, (batch_size,))
    force_value = floating(forces, (num_atoms, 3))
    force_valid = boolean(force_mask, (num_atoms, 3))
    force_available = boolean(force_present, (batch_size,))
    stress_value = floating(stress, (batch_size, 3, 3))
    stress_valid = boolean(stress_mask, (batch_size, 3, 3))
    stress_available = boolean(stress_present, (batch_size,))
    return StructureBatch(
        sample_ids=tuple(f"sample-{index}" for index in range(batch_size)),
        template_ids=("template",) * batch_size,
        template_fingerprints=("0" * 64,) * batch_size,
        positions=torch.zeros((num_atoms, 3), dtype=dtype, device=device),
        atomic_numbers=torch.full(
            (num_atoms,), 6, dtype=torch.long, device=device
        ),
        cells=torch.eye(3, dtype=dtype, device=device)
        .expand(batch_size, -1, -1)
        .clone(),
        origins=torch.zeros((batch_size, 3), dtype=dtype, device=device),
        pbc=torch.ones((batch_size, 3), dtype=torch.bool, device=device),
        atom_ptr=atom_ptr,
        atom_batch=atom_batch,
        energy=energy_value,
        energy_mask=energy_valid,
        forces=force_value,
        force_mask=force_valid,
        stress=stress_value,
        stress_mask=stress_valid,
        force_present=force_available,
        stress_present=stress_available,
        force_mask_provided=force_available.clone(),
        stress_mask_provided=stress_available.clone(),
    )


def _prediction(energy, *, forces=None, stress=None):
    return SimpleNamespace(energy=energy, forces=forces, stress=stress)


def test_manual_per_structure_and_per_atom_energy_loss():
    batch = _batch(
        (2, 4), energy=(0.0, 1.0), energy_mask=(True, True)
    )
    energy = torch.tensor([2.0, 5.0], dtype=torch.float64, requires_grad=True)
    prediction = _prediction(energy)

    per_structure = compute_potential_loss(
        prediction,
        batch,
        LossConfig(energy_scale=2.0, energy_normalization="per_structure"),
    )
    assert torch.equal(per_structure.energy.numerator, energy.new_tensor(5.0))
    assert torch.equal(per_structure.energy.denominator, energy.new_tensor(2.0))
    assert torch.equal(per_structure.energy.mean, energy.new_tensor(2.5))
    assert int(per_structure.energy.valid_count) == 2

    per_atom = compute_potential_loss(
        prediction,
        batch,
        LossConfig(energy_normalization="per_atom"),
    )
    assert torch.equal(per_atom.energy.numerator, energy.new_tensor(2.0))
    assert torch.equal(per_atom.energy.mean, energy.new_tensor(1.0))


def test_force_partial_components_and_zero_energy_label_are_distinct_from_missing():
    force_mask = torch.zeros((3, 3), dtype=torch.bool)
    force_mask[0, 0] = True
    force_mask[1, 2] = True
    batch = _batch(
        (2, 1),
        energy=(0.0, 0.0),
        energy_mask=(True, False),
        forces=torch.zeros((3, 3)),
        force_mask=force_mask,
        force_present=(True, False),
    )
    predicted_forces = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    predicted_forces = predicted_forces + torch.tensor(
        [[2.0, 9.0, 9.0], [9.0, 9.0, 4.0], [99.0, 99.0, 99.0]],
        dtype=torch.float64,
    )
    prediction = _prediction(
        torch.tensor([1.0, 7.0], dtype=torch.float64, requires_grad=True),
        forces=predicted_forces,
    )
    loss = compute_potential_loss(
        prediction,
        batch,
        LossConfig(energy_weight=1.0, force_weight=1.0, force_scale=2.0),
    )
    assert torch.equal(loss.energy.numerator, prediction.energy.new_tensor(1.0))
    assert int(loss.energy.valid_count) == 1
    assert torch.equal(loss.force.numerator, prediction.energy.new_tensor(5.0))
    assert torch.equal(loss.force.denominator, prediction.energy.new_tensor(2.0))
    assert torch.equal(loss.force.mean, prediction.energy.new_tensor(2.5))


def _symmetric_stress(dtype=torch.float64):
    return torch.tensor(
        [[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]],
        dtype=dtype,
    )


def test_stress_frobenius_factor_and_partial_symmetric_mask():
    full_batch = _batch(
        (1,),
        stress=torch.zeros((1, 3, 3)),
        stress_mask=torch.ones((1, 3, 3), dtype=torch.bool),
        stress_present=(True,),
    )
    stress = _symmetric_stress().unsqueeze(0).requires_grad_(True)
    prediction = _prediction(torch.zeros(1, dtype=torch.float64), stress=stress)
    full = compute_potential_loss(
        prediction,
        full_batch,
        LossConfig(energy_weight=0.0, stress_weight=1.0),
    )
    assert torch.equal(full.stress.numerator, stress.new_tensor(168.0))
    assert torch.equal(full.stress.denominator, stress.new_tensor(6.0))
    assert torch.equal(full.stress.mean, stress.new_tensor(28.0))

    mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = mask[0, 1, 0] = True
    partial_batch = _batch(
        (1,),
        stress=torch.zeros((1, 3, 3)),
        stress_mask=mask,
        stress_present=(True,),
    )
    partial = compute_potential_loss(
        prediction,
        partial_batch,
        LossConfig(energy_weight=0.0, stress_weight=1.0),
    )
    assert torch.equal(partial.stress.numerator, stress.new_tensor(33.0))
    assert torch.equal(partial.stress.denominator, stress.new_tensor(2.0))
    assert torch.equal(partial.stress.mean, stress.new_tensor(16.5))


def test_asymmetric_stress_mask_target_and_prediction_fail_fast():
    asymmetric_mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    asymmetric_mask[0, 0, 1] = True
    batch = _batch(
        (1,),
        stress=torch.zeros((1, 3, 3)),
        stress_mask=asymmetric_mask,
        stress_present=(True,),
    )
    prediction = _prediction(
        torch.zeros(1, dtype=torch.float64),
        stress=torch.zeros((1, 3, 3), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="asymmetric stress mask.*sample-0"):
        compute_potential_loss(
            prediction,
            batch,
            LossConfig(energy_weight=0.0, stress_weight=1.0),
        )

    target = torch.zeros((1, 3, 3), dtype=torch.float64)
    target[0, 0, 1] = 1.0
    target_batch = _batch(
        (1,),
        stress=target,
        stress_mask=torch.ones((1, 3, 3), dtype=torch.bool),
        stress_present=(True,),
    )
    with pytest.raises(ValueError, match="asymmetric stress target.*sample-0"):
        compute_potential_loss(
            prediction,
            target_batch,
            LossConfig(energy_weight=0.0, stress_weight=1.0),
        )

    predicted = torch.zeros((1, 3, 3), dtype=torch.float64)
    predicted[0, 1, 2] = 1.0
    with pytest.raises(ValueError, match="asymmetric stress prediction.*sample-0"):
        compute_potential_loss(
            _prediction(torch.zeros(1, dtype=torch.float64), stress=predicted),
            _batch(
                (1,),
                stress=torch.zeros((1, 3, 3)),
                stress_mask=torch.ones((1, 3, 3), dtype=torch.bool),
                stress_present=(True,),
            ),
            LossConfig(energy_weight=0.0, stress_weight=1.0),
        )


def test_zero_valid_terms_are_differentiable_and_zero_weight_needs_no_prediction():
    batch = _batch((2,))
    energy = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    loss = compute_potential_loss(
        _prediction(energy),
        batch,
        LossConfig(energy_weight=1.0, force_weight=1.0, stress_weight=1.0),
    )
    assert loss.total.requires_grad
    assert torch.equal(loss.total, energy.new_zeros(()))
    for term in (loss.energy, loss.force, loss.stress):
        assert torch.equal(term.numerator, energy.new_zeros(()))
        assert torch.equal(term.denominator, energy.new_zeros(()))
        assert int(term.valid_count) == 0
    loss.total.backward()
    assert torch.equal(energy.grad, torch.zeros_like(energy))

    nonfinite_missing = torch.tensor(
        [float("nan")], dtype=torch.float64, requires_grad=True
    )
    isolated = compute_potential_loss(
        _prediction(nonfinite_missing),
        batch,
        LossConfig(energy_weight=1.0, force_weight=1.0, stress_weight=1.0),
    )
    assert torch.equal(isolated.total, nonfinite_missing.new_zeros(()))
    isolated.total.backward()
    assert torch.equal(nonfinite_missing.grad, torch.zeros_like(nonfinite_missing))

    labeled = _batch(
        (1,),
        forces=torch.zeros((1, 3)),
        force_mask=torch.ones((1, 3), dtype=torch.bool),
        force_present=(True,),
        stress=torch.zeros((1, 3, 3)),
        stress_mask=torch.ones((1, 3, 3), dtype=torch.bool),
        stress_present=(True,),
    )
    ignored = compute_potential_loss(
        _prediction(torch.zeros(1, dtype=torch.float64, requires_grad=True)),
        labeled,
        LossConfig(energy_weight=0.0, force_weight=0.0, stress_weight=0.0),
    )
    assert torch.equal(ignored.total, torch.zeros((), dtype=torch.float64))


def test_weights_scales_and_force_stress_only_require_requested_predictions():
    force_mask = torch.zeros((1, 3), dtype=torch.bool)
    force_mask[0, 0] = True
    stress_mask = torch.zeros((1, 3, 3), dtype=torch.bool)
    stress_mask[0, 2, 2] = True
    batch = _batch(
        (1,),
        energy=(0.0,),
        energy_mask=(True,),
        forces=torch.zeros((1, 3)),
        force_mask=force_mask,
        force_present=(True,),
        stress=torch.zeros((1, 3, 3)),
        stress_mask=stress_mask,
        stress_present=(True,),
    )
    prediction = _prediction(
        torch.tensor([2.0], dtype=torch.float64),
        forces=torch.tensor([[4.0, 99.0, 99.0]], dtype=torch.float64),
        stress=torch.diag(torch.tensor([0.0, 0.0, 6.0], dtype=torch.float64)).unsqueeze(0),
    )
    config = LossConfig(
        energy_weight=2.0,
        force_weight=3.0,
        stress_weight=4.0,
        energy_scale=2.0,
        force_scale=2.0,
        stress_scale=3.0,
    )
    loss = compute_potential_loss(prediction, batch, config)
    assert torch.equal(loss.energy.mean, prediction.energy.new_tensor(1.0))
    assert torch.equal(loss.force.mean, prediction.energy.new_tensor(4.0))
    assert torch.equal(loss.stress.mean, prediction.energy.new_tensor(4.0))
    assert torch.equal(loss.total, prediction.energy.new_tensor(30.0))

    with pytest.raises(ValueError, match="force prediction.*sample-0"):
        compute_potential_loss(
            _prediction(prediction.energy),
            batch,
            LossConfig(energy_weight=0.0, force_weight=1.0),
        )
    with pytest.raises(ValueError, match="stress prediction.*sample-0"):
        compute_potential_loss(
            _prediction(prediction.energy),
            batch,
            LossConfig(energy_weight=0.0, stress_weight=1.0),
        )


def test_full_numerator_denominator_equal_split_statistics():
    full_batch = _batch(
        (1, 2),
        energy=(1.0, 2.0),
        energy_mask=(True, True),
        forces=torch.zeros((3, 3)),
        force_mask=torch.ones((3, 3), dtype=torch.bool),
        force_present=(True, True),
    )
    prediction = _prediction(
        torch.tensor([2.0, 4.0], dtype=torch.float64),
        forces=torch.arange(9, dtype=torch.float64).reshape(3, 3) / 10.0,
    )
    config = LossConfig(energy_weight=1.0, force_weight=1.0)
    full = compute_potential_loss(prediction, full_batch, config)

    first_batch = _batch(
        (1,),
        energy=(1.0,),
        energy_mask=(True,),
        forces=torch.zeros((1, 3)),
        force_mask=torch.ones((1, 3), dtype=torch.bool),
        force_present=(True,),
    )
    second_batch = _batch(
        (2,),
        energy=(2.0,),
        energy_mask=(True,),
        forces=torch.zeros((2, 3)),
        force_mask=torch.ones((2, 3), dtype=torch.bool),
        force_present=(True,),
    )
    first = compute_potential_loss(
        _prediction(prediction.energy[:1], forces=prediction.forces[:1]),
        first_batch,
        config,
    )
    second = compute_potential_loss(
        _prediction(prediction.energy[1:], forces=prediction.forces[1:]),
        second_batch,
        config,
    )
    for combined, left, right in (
        (full.energy, first.energy, second.energy),
        (full.force, first.force, second.force),
    ):
        assert torch.equal(combined.numerator, left.numerator + right.numerator)
        assert torch.equal(combined.denominator, left.denominator + right.denominator)


def test_structure_and_atom_permutation_parity():
    batch = _batch(
        (2, 1),
        energy=(1.0, 3.0),
        energy_mask=(True, True),
        forces=torch.zeros((3, 3)),
        force_mask=torch.ones((3, 3), dtype=torch.bool),
        force_present=(True, True),
    )
    energy = torch.tensor([2.0, 5.0], dtype=torch.float64)
    forces = torch.arange(9, dtype=torch.float64).reshape(3, 3)
    config = LossConfig(energy_weight=1.0, force_weight=1.0)
    original = compute_potential_loss(
        _prediction(energy, forces=forces), batch, config
    )

    atom_order = torch.tensor([2, 1, 0])
    permuted_batch = _batch(
        (1, 2),
        energy=(3.0, 1.0),
        energy_mask=(True, True),
        forces=torch.zeros((3, 3)),
        force_mask=torch.ones((3, 3), dtype=torch.bool),
        force_present=(True, True),
    )
    permuted = compute_potential_loss(
        _prediction(energy.flip(0), forces=forces[atom_order]),
        permuted_batch,
        config,
    )
    assert torch.equal(permuted.total, original.total)
    assert torch.equal(permuted.energy.numerator, original.energy.numerator)
    assert torch.equal(permuted.force.numerator, original.force.numerator)


def test_config_validation_dtype_shape_and_nonfinite_fail_fast():
    config = LossConfig(
        energy_weight=2.0,
        force_scale=3.0,
        energy_normalization="per_atom",
    )
    assert LossConfig.from_dict(config.to_dict()) == config
    with pytest.raises((TypeError, ValueError)):
        LossConfig(energy_weight=True)
    with pytest.raises(ValueError):
        LossConfig(force_scale=0.0)
    with pytest.raises(ValueError):
        LossConfig(stress_symmetry_tolerance=float("nan"))
    with pytest.raises(ValueError):
        LossConfig(energy_normalization="invalid")

    batch = _batch((1,), energy=(0.0,), energy_mask=(True,))
    with pytest.raises(ValueError, match="dtype"):
        compute_potential_loss(
            _prediction(torch.zeros(1, dtype=torch.float32)), batch, LossConfig()
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_potential_loss(
            _prediction(torch.zeros(2, dtype=torch.float64)), batch, LossConfig()
        )
    with pytest.raises(ValueError, match="nonfinite energy prediction.*sample-0"):
        compute_potential_loss(
            _prediction(torch.tensor([float("nan")], dtype=torch.float64)),
            batch,
            LossConfig(),
        )


def test_cpu_float64_backward():
    batch = _batch(
        (2,),
        energy=(0.0,),
        energy_mask=(True,),
        forces=torch.zeros((2, 3)),
        force_mask=torch.ones((2, 3), dtype=torch.bool),
        force_present=(True,),
    )
    energy = torch.ones(1, dtype=torch.float64, requires_grad=True)
    forces = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
    loss = compute_potential_loss(
        _prediction(energy, forces=forces),
        batch,
        LossConfig(energy_weight=1.0, force_weight=1.0),
    )
    loss.total.backward()
    assert bool(torch.all(torch.isfinite(energy.grad)))
    assert bool(torch.all(torch.isfinite(forces.grad)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_smoke_when_available(dtype):
    if not torch.cuda.is_available():
        return
    batch = _batch(
        (1,),
        dtype=dtype,
        device="cuda",
        energy=(0.0,),
        energy_mask=(True,),
        forces=torch.zeros((1, 3)),
        force_mask=torch.ones((1, 3), dtype=torch.bool),
        force_present=(True,),
        stress=torch.zeros((1, 3, 3)),
        stress_mask=torch.ones((1, 3, 3), dtype=torch.bool),
        stress_present=(True,),
    )
    energy = torch.ones(1, dtype=dtype, device="cuda", requires_grad=True)
    forces = torch.ones((1, 3), dtype=dtype, device="cuda", requires_grad=True)
    stress = torch.eye(3, dtype=dtype, device="cuda").unsqueeze(0).requires_grad_(True)
    with pytest.raises(ValueError, match="device"):
        compute_potential_loss(
            _prediction(torch.ones(1, dtype=dtype)), batch, LossConfig()
        )
    loss = compute_potential_loss(
        _prediction(energy, forces=forces, stress=stress),
        batch,
        LossConfig(energy_weight=1.0, force_weight=1.0, stress_weight=1.0),
    )
    loss.total.backward()
    assert loss.total.dtype == dtype and loss.total.device.type == "cuda"
    assert bool(torch.isfinite(loss.total))
    assert bool(torch.all(torch.isfinite(energy.grad)))
    assert bool(torch.all(torch.isfinite(forces.grad)))
    assert bool(torch.all(torch.isfinite(stress.grad)))
