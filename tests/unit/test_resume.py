from __future__ import annotations

import copy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from refsite_mlip.data import StructureBatch
from refsite_mlip.training import (
    CheckpointCompatibilityError,
    CheckpointRestoreError,
    FitConfig,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    ResumePolicy,
    ResumeState,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_scheduler,
    capture_training_checkpoint,
    restore_training_checkpoint_,
    validate_checkpoint_compatibility,
)


resume_module = __import__("refsite_mlip.training.resume", fromlist=["resume"])


class TinyModel(torch.nn.Module):
    def __init__(self, *, width=2, dtype=torch.float64):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(-0.5, 1.0, width, dtype=dtype)
        )
        self.register_buffer(
            "atomic_baseline", torch.tensor([0.25], dtype=dtype)
        )
        self.config = SimpleNamespace(species_vocabulary=(6,))


def _batch(*, split="train", shift=0.0, fingerprint="2" * 64):
    return StructureBatch(
        sample_ids=(split,),
        template_ids=("template",),
        template_fingerprints=(fingerprint,),
        positions=torch.tensor([[0.1 + shift, 0.2, 0.3]], dtype=torch.float64),
        atomic_numbers=torch.tensor([6], dtype=torch.long),
        cells=torch.eye(3, dtype=torch.float64).reshape(1, 3, 3) * 4.0,
        origins=torch.zeros((1, 3), dtype=torch.float64),
        pbc=torch.ones((1, 3), dtype=torch.bool),
        atom_ptr=torch.tensor([0, 1], dtype=torch.long),
        atom_batch=torch.zeros(1, dtype=torch.long),
        energy=torch.tensor([1.0], dtype=torch.float64),
        energy_mask=torch.ones(1, dtype=torch.bool),
        forces=torch.zeros((1, 3), dtype=torch.float64),
        force_mask=torch.ones((1, 3), dtype=torch.bool),
        stress=torch.zeros((1, 3, 3), dtype=torch.float64),
        stress_mask=torch.ones((1, 3, 3), dtype=torch.bool),
        force_present=torch.ones(1, dtype=torch.bool),
        stress_present=torch.ones(1, dtype=torch.bool),
        force_mask_provided=torch.ones(1, dtype=torch.bool),
        stress_mask_provided=torch.ones(1, dtype=torch.bool),
    )


def _live():
    model = TinyModel()
    optimizer_config = OptimizerConfig(learning_rate=0.01, weight_decay=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config.learning_rate, weight_decay=0.0
    )
    optimizer.zero_grad(set_to_none=True)
    model.weight.square().sum().backward()
    optimizer.step()
    scheduler_config = SchedulerConfig(kind="reduce_on_plateau", patience=0)
    scheduler = build_scheduler(optimizer, scheduler_config)
    scheduler.step(1.0)
    selection = ModelSelectionState(
        best_metric=1.0,
        best_epoch=1,
        best_global_step=3,
        validation_events=2,
        last_validation_epoch=1,
        last_validation_global_step=3,
    )
    progress = FitProgress(
        next_epoch=2,
        global_step=3,
        completed_epochs=2,
        last_completed_epoch=1,
        best_epoch=1,
        best_global_step=3,
    )
    return (
        model,
        optimizer,
        scheduler,
        scheduler_config,
        optimizer_config,
        selection,
        progress,
    )


def _case(monkeypatch):
    train = (_batch(split="train"),)
    validation = (_batch(split="validation"),)
    (
        model,
        optimizer,
        scheduler,
        scheduler_config,
        optimizer_config,
        selection,
        progress,
    ) = _live()
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        selection,
        progress,
        train,
        validation,
        model_config={"hidden": 2, "convention_version": "tiny_v1"},
        loss_config=LossConfig(),
        optimizer_config=optimizer_config,
        train_step_config=TrainStepConfig(),
        validation_step_config=ValidationStepConfig(),
        scheduler_config=scheduler_config,
        model_selection_config=ModelSelectionConfig(),
        fit_config=FitConfig(3),
        species_vocabulary=(6,),
        baseline_fit_metadata={"kind": "explicit", "values": [0.25]},
        source_git_commit="saved-git",
    )
    target, target_optimizer, target_scheduler, *_ = _live()
    with torch.no_grad():
        target.weight.add_(5.0)
        target.atomic_baseline.fill_(-2.0)
    target_optimizer.param_groups[0]["lr"] = 0.123
    target_scheduler.step(3.0)
    target.weight.grad = torch.tensor([7.0, 8.0], dtype=torch.float64)
    resolved = copy.deepcopy(checkpoint.metadata.resolved_configuration)
    resolved["fit"] = {**resolved["fit"], "max_epochs": 4}
    resolved["baseline_fit_metadata"] = copy.deepcopy(
        checkpoint.metadata.baseline_fit_metadata
    )
    contexts = {"template": object()}
    monkeypatch.setattr(
        resume_module,
        "_validated_context",
        lambda template_id, fingerprint, mapping: mapping[template_id],
    )
    return {
        "checkpoint": checkpoint,
        "model": target,
        "optimizer": target_optimizer,
        "scheduler": target_scheduler,
        "train": train,
        "validation": validation,
        "contexts": contexts,
        "resolved": resolved,
    }


def _call(case, function=restore_training_checkpoint_, **overrides):
    values = dict(
        checkpoint=case["checkpoint"],
        model=case["model"],
        optimizer=case["optimizer"],
        scheduler=case["scheduler"],
        train_batches=case["train"],
        validation_batches=case["validation"],
        template_contexts=case["contexts"],
        resolved_configs=case["resolved"],
        resumed_max_epochs=4,
        policy=ResumePolicy(),
        current_source_git_commit="different-git",
    )
    values.update(overrides)
    return function(**values)


def _tree_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _tree_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_clone(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _tree_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            _tree_equal(a, b) for a, b in zip(first, second)
        )
    return first == second


def _rng():
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        torch.get_rng_state().clone(),
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    )


def _rng_equal(first, second):
    return (
        first[0] == second[0]
        and first[1][0] == second[1][0]
        and np.array_equal(first[1][1], second[1][1])
        and first[1][2:] == second[1][2:]
        and torch.equal(first[2], second[2])
        and len(first[3]) == len(second[3])
        and all(torch.equal(a, b) for a, b in zip(first[3], second[3]))
    )


def test_policy_and_resume_state_serialization(monkeypatch):
    policy = ResumePolicy(require_git_commit_match=True, restore_cuda_rng=False)
    assert ResumePolicy.from_dict(policy.to_dict()) == policy
    case = _case(monkeypatch)
    result = _call(case)
    assert ResumeState.from_dict(result.to_dict()) == result
    assert result.next_epoch == 2 and result.global_step == 3
    assert result.resumed_fit_config == FitConfig(4, 2, 3)
    assert result.exact_resume_ready


def test_compatibility_validation_is_completely_read_only(monkeypatch):
    case = _case(monkeypatch)
    model_state = _tree_clone(case["model"].state_dict())
    optimizer_state = _tree_clone(case["optimizer"].state_dict())
    scheduler_state = _tree_clone(case["scheduler"].state_dict())
    gradient = case["model"].weight.grad
    gradient_value = gradient.clone()
    rng = _rng()
    checkpoint_payload = _tree_clone(case["checkpoint"].to_dict())
    diagnostics = _call(case, function=validate_checkpoint_compatibility)
    assert diagnostics
    assert _tree_equal(model_state, case["model"].state_dict())
    assert _tree_equal(optimizer_state, case["optimizer"].state_dict())
    assert _tree_equal(scheduler_state, case["scheduler"].state_dict())
    assert case["model"].weight.grad is gradient
    assert torch.equal(case["model"].weight.grad, gradient_value)
    assert _rng_equal(rng, _rng())
    assert _tree_equal(checkpoint_payload, case["checkpoint"].to_dict())


def test_strict_transactional_restore_preserves_parameter_identity(monkeypatch):
    case = _case(monkeypatch)
    parameter_ids = tuple(id(parameter) for parameter in case["model"].parameters())
    checkpoint_payload = _tree_clone(case["checkpoint"].to_dict())
    original_mode = case["model"].training
    result = _call(case)
    assert _tree_equal(case["model"].state_dict(), case["checkpoint"].model_state_dict)
    assert _tree_equal(
        case["optimizer"].state_dict(), case["checkpoint"].optimizer_state_dict
    )
    assert _tree_equal(
        case["scheduler"].state_dict(), case["checkpoint"].scheduler_state_dict
    )
    assert tuple(id(parameter) for parameter in case["model"].parameters()) == parameter_ids
    assert tuple(case["optimizer"].param_groups[0]["params"]) == tuple(
        case["model"].parameters()
    )
    assert case["scheduler"].optimizer is case["optimizer"]
    assert all(parameter.grad is None for parameter in case["model"].parameters())
    assert case["model"].training is original_mode
    assert torch.equal(
        case["model"].atomic_baseline,
        case["checkpoint"].model_state_dict["atomic_baseline"],
    )
    assert result.selection_state == case["checkpoint"].selection_state
    assert _tree_equal(checkpoint_payload, case["checkpoint"].to_dict())


def test_python_numpy_and_torch_next_draw_rng_parity(monkeypatch):
    case = _case(monkeypatch)
    saved = _rng()
    resume_module._restore_checkpoint_rng(case["checkpoint"], ResumePolicy())
    expected = (random.random(), float(np.random.random()), torch.rand(4))
    resume_module._restore_raw_rng(
        {"python": saved[0], "numpy": saved[1], "torch_cpu": saved[2], "cuda": saved[3]}
    )
    _call(case)
    actual = (random.random(), float(np.random.random()), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_disabled_rng_domain_marks_resume_not_exact(monkeypatch):
    case = _case(monkeypatch)
    result = _call(
        case,
        policy=ResumePolicy(restore_numpy_rng=False),
    )
    assert not result.exact_resume_ready
    assert "numpy" not in result.restored_rng_domains


def test_all_cuda_rng_next_draw_parity_when_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    case = _case(monkeypatch)
    saved = _rng()
    resume_module._restore_checkpoint_rng(case["checkpoint"], ResumePolicy())
    expected = tuple(
        torch.rand(4, device=f"cuda:{index}").cpu()
        for index in range(torch.cuda.device_count())
    )
    resume_module._restore_raw_rng(
        {"python": saved[0], "numpy": saved[1], "torch_cpu": saved[2], "cuda": saved[3]}
    )
    _call(case)
    actual = tuple(
        torch.rand(4, device=f"cuda:{index}").cpu()
        for index in range(torch.cuda.device_count())
    )
    assert len(actual) == len(expected)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("data", "training data manifest"),
        ("species", "species vocabulary"),
        ("units", "unit/stress/Voigt"),
        ("config", "loss configuration"),
        ("baseline", "baseline fit metadata"),
        ("max_decrease", "max_epochs"),
    ],
)
def test_compatibility_mismatches_fail_before_mutation(monkeypatch, mutation, match):
    case = _case(monkeypatch)
    kwargs = {}
    if mutation == "data":
        kwargs["train_batches"] = (_batch(split="train", shift=0.1),)
    elif mutation == "species":
        case["model"].config.species_vocabulary = (8,)
    elif mutation == "units":
        case["resolved"]["unit_conventions"] = {
            **case["checkpoint"].metadata.unit_conventions,
            "stress_sign": "compressive_positive",
        }
    elif mutation == "config":
        case["resolved"]["loss"] = {
            **case["resolved"]["loss"],
            "force_weight": 3.0,
        }
    elif mutation == "baseline":
        case["resolved"]["baseline_fit_metadata"] = {"kind": "other"}
    elif mutation == "max_decrease":
        case["resolved"]["fit"]["max_epochs"] = 2
        kwargs["resumed_max_epochs"] = 2
    before = _tree_clone(case["model"].state_dict())
    rng = _rng()
    with pytest.raises(CheckpointCompatibilityError, match=match):
        _call(case, function=validate_checkpoint_compatibility, **kwargs)
    assert _tree_equal(before, case["model"].state_dict())
    assert _rng_equal(rng, _rng())


def test_git_and_version_policy(monkeypatch):
    case = _case(monkeypatch)
    with pytest.raises(CheckpointCompatibilityError, match="source git commit"):
        _call(
            case,
            function=validate_checkpoint_compatibility,
            policy=ResumePolicy(require_git_commit_match=True),
        )
    metadata = replace(
        case["checkpoint"].metadata,
        package_versions={**case["checkpoint"].metadata.package_versions, "torch": "0"},
    )
    case["checkpoint"] = replace(case["checkpoint"], metadata=metadata)
    with pytest.raises(CheckpointCompatibilityError, match="package versions"):
        _call(case, function=validate_checkpoint_compatibility)
    assert _call(
        case,
        function=validate_checkpoint_compatibility,
        policy=ResumePolicy(require_version_match=False),
    )


def test_model_optimizer_scheduler_and_stopped_state_rejection(monkeypatch):
    case = _case(monkeypatch)
    wrong_model = TinyModel(width=3)
    wrong_optimizer = torch.optim.AdamW(wrong_model.parameters(), lr=0.01)
    wrong_scheduler = build_scheduler(
        wrong_optimizer, SchedulerConfig(kind="reduce_on_plateau", patience=0)
    )
    with pytest.raises(CheckpointCompatibilityError, match="shape mismatch"):
        _call(
            case,
            function=validate_checkpoint_compatibility,
            model=wrong_model,
            optimizer=wrong_optimizer,
            scheduler=wrong_scheduler,
        )
    dtype_model = TinyModel(dtype=torch.float32)
    dtype_optimizer = torch.optim.AdamW(dtype_model.parameters(), lr=0.01)
    dtype_scheduler = build_scheduler(
        dtype_optimizer, SchedulerConfig(kind="reduce_on_plateau", patience=0)
    )
    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        _call(
            case,
            function=validate_checkpoint_compatibility,
            model=dtype_model,
            optimizer=dtype_optimizer,
            scheduler=dtype_scheduler,
        )
    other_optimizer = torch.optim.AdamW(case["model"].parameters(), lr=0.01)
    with pytest.raises(CheckpointCompatibilityError, match="scheduler"):
        _call(
            case,
            function=validate_checkpoint_compatibility,
            optimizer=other_optimizer,
        )
    stopped = ModelSelectionState(
        best_metric=1.0,
        best_epoch=1,
        best_global_step=3,
        validation_events=2,
        last_validation_epoch=1,
        last_validation_global_step=3,
        stopped_early=True,
        stop_epoch=1,
        stop_reason="patience",
    )
    case["checkpoint"] = replace(
        case["checkpoint"],
        selection_state=stopped,
        progress=replace(case["checkpoint"].progress, stopped_early=True),
    )
    with pytest.raises(CheckpointCompatibilityError, match="already stopped"):
        _call(case, function=validate_checkpoint_compatibility)


@pytest.mark.parametrize("failure_stage", ["model", "optimizer", "scheduler", "rng"])
def test_failure_rolls_back_all_state_grad_mode_and_rng(monkeypatch, failure_stage):
    case = _case(monkeypatch)
    model_state = _tree_clone(case["model"].state_dict())
    optimizer_state = _tree_clone(case["optimizer"].state_dict())
    scheduler_state = _tree_clone(case["scheduler"].state_dict())
    gradient = case["model"].weight.grad
    gradient_value = gradient.clone()
    mode = case["model"].training
    rng = _rng()
    if failure_stage == "model":
        original = case["model"].load_state_dict
        calls = {"count": 0}
        def fail_once(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("injected model failure")
            return original(*args, **kwargs)
        monkeypatch.setattr(case["model"], "load_state_dict", fail_once)
    elif failure_stage == "optimizer":
        original = case["optimizer"].load_state_dict
        calls = {"count": 0}
        def fail_once(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("injected optimizer failure")
            return original(*args, **kwargs)
        monkeypatch.setattr(case["optimizer"], "load_state_dict", fail_once)
    elif failure_stage == "scheduler":
        scheduler_type = type(case["scheduler"])
        original = scheduler_type.load_state_dict
        calls = {"count": 0}
        def fail_once(instance, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("injected scheduler failure")
            return original(instance, *args, **kwargs)
        monkeypatch.setattr(scheduler_type, "load_state_dict", fail_once)
    else:
        original = resume_module._restore_checkpoint_rng
        calls = {"count": 0}
        def fail_rng(*args, **kwargs):
            calls["count"] += 1
            random.random()
            raise RuntimeError("injected RNG failure")
        monkeypatch.setattr(resume_module, "_restore_checkpoint_rng", fail_rng)
    with pytest.raises(CheckpointRestoreError, match="rollback succeeded") as caught:
        _call(case)
    assert caught.value.rollback_succeeded
    assert _tree_equal(model_state, case["model"].state_dict())
    assert _tree_equal(optimizer_state, case["optimizer"].state_dict())
    assert _tree_equal(scheduler_state, case["scheduler"].state_dict())
    assert case["model"].weight.grad is gradient
    assert torch.equal(case["model"].weight.grad, gradient_value)
    assert case["model"].training is mode
    assert _rng_equal(rng, _rng())


def test_context_and_fingerprint_mismatch_fail_fast(monkeypatch):
    case = _case(monkeypatch)
    case["contexts"] = {}
    with pytest.raises(CheckpointCompatibilityError, match="context ID set"):
        _call(case, function=validate_checkpoint_compatibility)
    case = _case(monkeypatch)
    case["train"] = (_batch(split="train", fingerprint="3" * 64),)
    with pytest.raises(CheckpointCompatibilityError, match="training data manifest"):
        _call(case, function=validate_checkpoint_compatibility)
