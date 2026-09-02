from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math

import pytest

from refsite_mlip.config import (
    TRAINING_RUN_CONFIG_SCHEMA_VERSION,
    InteractionRadiusConfig,
    TrainingDataConfig,
    TrainingRuntimeConfig,
    TrainingRunConfig,
    TrainingRunConfigError,
    validate_training_run_config,
)
from refsite_mlip.training import (
    AtomicBaselineConfig,
    CheckpointedFitConfig,
    FitConfig,
    LossConfig,
    ModelSelectionConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
)


def _payload() -> dict:
    return {
        "schema_version": TRAINING_RUN_CONFIG_SCHEMA_VERSION,
        "initial_bundle": "initial-model.pt",
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "data": {
            "train": [{"path": "train.xyz", "template_id": "template-a"}],
            "validation": [
                {"path": "validation.xyz", "template_key": "template"}
            ],
            "batch_size": 4,
            "shuffle": False,
        },
        "runtime": {"device": "cpu", "dtype": "float64"},
        "loss": LossConfig().to_dict(),
        "baseline": AtomicBaselineConfig().to_dict(),
        "optimizer": OptimizerConfig().to_dict(),
        "train_step": TrainStepConfig().to_dict(),
        "validation_step": ValidationStepConfig().to_dict(),
        "scheduler": SchedulerConfig().to_dict(),
        "selection": ModelSelectionConfig().to_dict(),
        "fit": FitConfig(max_epochs=3).to_dict(),
        "checkpointed_fit": CheckpointedFitConfig().to_dict(),
        "output_directory": "run-output",
    }


def _assert_plain(value):
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_plain(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain(item)
        return
    assert value is None or type(value) in (str, bool, int, float)
    if type(value) is float:
        assert math.isfinite(value)


def test_minimal_json_round_trip_is_canonical_plain_and_cwd_independent():
    first = TrainingRunConfig.from_dict(_payload())
    reversed_payload = dict(reversed(tuple(_payload().items())))
    second = TrainingRunConfig.from_dict(reversed_payload)

    assert first == second
    assert first.radii == InteractionRadiusConfig()
    assert first.canonical_json() == second.canonical_json()
    assert first.config_fingerprint == second.config_fingerprint
    assert first.content_fingerprint == first.config_fingerprint
    assert first.fingerprint == hashlib.sha256(
        first.canonical_json().encode("utf-8")
    ).hexdigest()
    assert TrainingRunConfig.from_json(first.to_json()) == first
    _assert_plain(first.to_dict())

    relocated = replace(first, source_path="/unrelated/location/run.json")
    assert relocated.config_fingerprint == first.config_fingerprint
    assert relocated.to_dict() == first.to_dict()
    assert first.to_dict()["initial_bundle"] == "initial-model.pt"
    assert first.to_dict()["data"]["train"][0]["path"] == "train.xyz"


def test_full_config_reuses_every_existing_config_serialization():
    payload = _payload()
    configs = {
        "loss": LossConfig(
            energy_weight=2.0,
            force_weight=3.0,
            stress_weight=4.0,
            energy_scale=5.0,
            force_scale=6.0,
            stress_scale=7.0,
            energy_normalization="per_atom",
        ),
        "baseline": AtomicBaselineConfig(
            weighting="per_atom", rcond=1.0e-10, ridge=0.2,
            rank_policy="minimum_norm",
        ),
        "optimizer": OptimizerConfig(
            learning_rate=2.0e-4,
            betas=(0.8, 0.95),
            eps=1.0e-7,
            weight_decay=0.01,
            amsgrad=True,
        ),
        "train_step": TrainStepConfig(gradient_clip_norm=2.5),
        "validation_step": ValidationStepConfig(),
        "scheduler": SchedulerConfig(
            kind="reduce_on_plateau", monitor="stress", mode="max", patience=2
        ),
        "selection": ModelSelectionConfig(
            monitor="stress", mode="max", min_delta=0.3,
            early_stopping_patience=4,
        ),
        "fit": FitConfig(max_epochs=8),
        "checkpointed_fit": CheckpointedFitConfig(),
    }
    payload["radii"] = InteractionRadiusConfig(
        r_ot=5.0,
        r_mp=2.5,
        ot_switch_width=0.75,
        ot_skin=0.4,
        mp_skin=0.25,
    ).to_dict()
    for name, config in configs.items():
        payload[name] = config.to_dict()

    parsed = TrainingRunConfig.from_dict(payload)
    for name, config in configs.items():
        assert getattr(parsed, name) == config
        assert parsed.to_dict()[name] == config.to_dict()
        assert type(config).from_dict(parsed.to_dict()[name]) == config
    assert TrainingRunConfig.from_dict(parsed.to_dict()) == parsed


def test_public_data_and_runtime_configs_are_frozen_and_strict():
    data = TrainingDataConfig(
        train=({"path": "a.xyz", "template_id": "a"},),
        validation=({"path": "b.xyz", "template_key": "template"},),
        batch_size=2,
        shuffle=False,
    )
    assert data.to_dict()["train"] == [{"path": "a.xyz", "template_id": "a"}]
    runtime = TrainingRuntimeConfig(device="cuda:2", dtype="float32")
    assert runtime.to_dict() == {"device": "cuda:2", "dtype": "float32"}
    with pytest.raises(FrozenInstanceError):
        runtime.device = "cpu"
    with pytest.raises(TrainingRunConfigError, match="shuffle=false"):
        replace(data, shuffle=True)
    with pytest.raises(TrainingRunConfigError, match="exactly one"):
        TrainingDataConfig(
            train=({"path": "a.xyz", "template_id": "a", "template_key": "k"},),
            validation=data.validation,
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value.update({"unknown": 1}), "UNKNOWN_CONFIG_KEY"),
        (lambda value: value.pop("initial_bundle"), "MISSING_CONFIG_KEY"),
        (
            lambda value: value["optimizer"].pop("eps"),
            "MISSING_CONFIG_KEY",
        ),
        (
            lambda value: value["data"].update({"batch_size": True}),
            "INVALID_BATCH_SIZE",
        ),
        (
            lambda value: value["loss"].update({"energy_weight": True}),
            "INVALID_TRAINING_CONFIG",
        ),
        (
            lambda value: value["radii"].update({"r_ot": False}),
            "INVALID_RADIUS_VALUE",
        ),
        (
            lambda value: value["runtime"].update({"dtype": False}),
            "INVALID_RUNTIME_DTYPE",
        ),
        (
            lambda value: value["data"].update({"shuffle": True}),
            "UNSUPPORTED_SHUFFLE",
        ),
    ],
)
def test_unknown_missing_bool_and_unsupported_values_are_structured(mutate, reason):
    payload = _payload()
    mutate(payload)
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == reason


def test_duplicate_nonfinite_and_invalid_json_are_rejected():
    encoded = json.dumps(_payload(), sort_keys=True)
    duplicate = encoded.replace(
        '"initial_bundle": "initial-model.pt"',
        '"initial_bundle": "initial-model.pt", "initial_bundle": "other.pt"',
    )
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_json(duplicate)
    assert caught.value.reason_code == "CONFLICTING_CONFIG_KEY"

    nonfinite_payload = _payload()
    nonfinite_payload["loss"]["energy_scale"] = float("nan")
    nonfinite = json.dumps(nonfinite_payload, allow_nan=True)
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_json(nonfinite)
    assert caught.value.reason_code == "NONFINITE_CONFIG_VALUE"

    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_json("[]")
    assert caught.value.reason_code == "INVALID_CONFIG_SECTION"


def test_solver_monitor_mode_and_fresh_fit_cross_validation():
    payload = _payload()
    payload["train_step"]["solver_path"] = "eval_adaptive"
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "UNSUPPORTED_TRAINING_SOLVER"
    assert caught.value.field == "train_step.solver_path"

    payload = _payload()
    payload["scheduler"]["monitor"] = "energy"
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "MONITOR_MISMATCH"

    payload = _payload()
    payload["scheduler"]["mode"] = "max"
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "MONITOR_MODE_MISMATCH"

    config = TrainingRunConfig.from_dict(_payload())
    invalid = object.__new__(TrainingRunConfig)
    # Cross validation remains callable independently of construction.
    for field_name in config.__dataclass_fields__:
        object.__setattr__(invalid, field_name, getattr(config, field_name))
    object.__setattr__(invalid, "fit", FitConfig(max_epochs=3, start_epoch=1))
    with pytest.raises(TrainingRunConfigError) as caught:
        validate_training_run_config(invalid)
    assert caught.value.reason_code == "FRESH_RUN_PROGRESS_REQUIRED"


def test_path_text_is_not_environment_or_shell_expanded():
    payload = _payload()
    payload["initial_bundle"] = "$HOME/bundle.pt"
    payload["data"]["train"][0]["path"] = "~/train.xyz"
    payload["output_directory"] = "$(pwd)/output"
    config = TrainingRunConfig.from_dict(payload)
    assert config.initial_bundle == "$HOME/bundle.pt"
    assert config.data.train[0].path == "~/train.xyz"
    assert config.output_directory == "$(pwd)/output"
