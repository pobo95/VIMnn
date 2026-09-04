from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import random

import numpy as np
import pytest
import torch
import yaml

from refsite_mlip.config import (
    TRAINING_RUN_CONFIG_SCHEMA_VERSION,
    TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2,
    BundleModelSourceConfig,
    InteractionRadiusConfig,
    ScratchModelSourceConfig,
    TrainingDataConfig,
    TrainingRunConfigOverrides,
    TrainingRuntimeConfig,
    TrainingRunConfig,
    TrainingRunConfigError,
    apply_training_run_overrides,
    load_training_run_config,
    resolve_training_run,
    validate_training_run_config,
)
from refsite_mlip.data import (
    PhaseSpecification,
    nbc_rocksalt_template_builder_config,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import PotentialConfig
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
from refsite_mlip.transport import TransportSupportConfig


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
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 17},
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


def _scratch_source_payload() -> dict:
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=2,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        probability_tolerance=1.0e-8,
        site_type_vocabulary=(0, 1),
    )
    higher = HigherBodyConfig(
        irreps_feature="2x0e+4x0e+4x1o+4x2e",
        species_count=2,
        site_type_count=2,
        site_type_embedding_dim=2,
        n_correlation_channels=1,
        lmax=2,
        radial_feature_dim=3,
        radial_hidden_dims=(4,),
        avg_num_neighbors=6.0,
        cutoff=3.0,
        edge_length_scale=1.0,
    )
    potential = PotentialConfig(
        species_vocabulary=(6, 41),
        num_layers=1,
        feature=feature,
        higher_body=higher,
        transport_support=TransportSupportConfig(
            kind="compact_c2",
            cutoff=4.0,
            switch_width=0.5,
            candidate_skin=0.2,
        ),
    )
    builder = nbc_rocksalt_template_builder_config((2, 2, 2))
    phase = PhaseSpecification(
        modes=torch.eye(3, dtype=torch.long),
        mode_weights=torch.ones(3, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status="provisional",
    )
    return {
        "kind": "scratch",
        "initialization_seed": 20260904,
        "potential": potential.to_dict(),
        "species_alignment_weights": [[1.0, -0.5], [-1.0, 2.0]],
        "reference_templates": [
            {
                "poscar_path": "references/NbC.POSCAR",
                "builder": builder.to_dict(),
                "phase_specification": phase.to_dict(),
                "evaluation_policy": None,
            }
        ],
        "default_template_id": builder.template_id,
    }


def _v2_payload(*, kind: str = "scratch") -> dict:
    payload = _payload()
    payload["schema_version"] = TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2
    bundle = payload.pop("initial_bundle")
    payload["data"]["validation_batch_size"] = 2
    payload["model_source"] = (
        {"kind": "bundle", "path": bundle}
        if kind == "bundle"
        else _scratch_source_payload()
    )
    return payload


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

    payload["baseline"] = None
    disabled = TrainingRunConfig.from_dict(payload)
    assert disabled.baseline is None
    assert disabled.to_dict()["baseline"] is None


def test_public_data_and_runtime_configs_are_frozen_and_strict():
    data = TrainingDataConfig(
        train=({"path": "a.xyz", "template_id": "a"},),
        validation=({"path": "b.xyz", "template_key": "template"},),
        batch_size=2,
        shuffle=False,
    )
    assert data.to_dict()["train"] == [{"path": "a.xyz", "template_id": "a"}]
    runtime = TrainingRuntimeConfig(seed=23, device="cuda:2", dtype="float32")
    assert runtime.to_dict() == {
        "device": "cuda:2",
        "dtype": "float32",
        "seed": 23,
    }
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
            lambda value: value["runtime"].update({"seed": False}),
            "INVALID_RUNTIME_SEED",
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

    missing_seed = _payload()
    missing_seed["runtime"].pop("seed")
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(missing_seed)
    assert caught.value.reason_code == "MISSING_CONFIG_KEY"
    assert "seed" in (caught.value.field or "")


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


def test_v1_source_normalization_preserves_exact_payload_and_fingerprint():
    payload = _payload()
    config = TrainingRunConfig.from_dict(payload)
    assert isinstance(config.model_source, BundleModelSourceConfig)
    assert config.model_source.path == payload["initial_bundle"]
    canonical_payload = dict(payload)
    canonical_payload["radii"] = InteractionRadiusConfig().to_dict()
    assert config.to_dict() == canonical_payload
    expected = hashlib.sha256(
        json.dumps(
            canonical_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert config.config_fingerprint == expected


@pytest.mark.parametrize("kind", ["bundle", "scratch"])
def test_v2_tagged_model_source_round_trip_is_canonical(kind):
    config = TrainingRunConfig.from_dict(_v2_payload(kind=kind))
    if kind == "bundle":
        assert isinstance(config.model_source, BundleModelSourceConfig)
        assert config.initial_bundle == "initial-model.pt"
    else:
        assert isinstance(config.model_source, ScratchModelSourceConfig)
        assert config.initial_bundle is None
        template = config.model_source.reference_templates[0]
        assert template.template_id == "nbc_rocksalt_222_v1"
        assert template.builder.strict_domain.supercell_shape == (2, 2, 2)
        assert template.builder.site_type_ids == (0, 1)
        assert template.poscar_path == "references/NbC.POSCAR"
    assert config.data.effective_validation_batch_size == 2
    assert TrainingRunConfig.from_dict(config.to_dict()) == config
    _assert_plain(config.to_dict())


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda value: value.update(initial_bundle="also.pt"),
            "CONFLICTING_MODEL_SOURCE",
        ),
        (
            lambda value: value.pop("model_source"),
            "MISSING_MODEL_SOURCE",
        ),
        (
            lambda value: value["model_source"].update(kind="resume"),
            "INVALID_MODEL_SOURCE_KIND",
        ),
        (
            lambda value: value["model_source"].update(extra=True),
            "UNKNOWN_CONFIG_KEY",
        ),
        (
            lambda value: value["model_source"]["reference_templates"][0][
                "builder"
            ].update(extra=True),
            "UNKNOWN_CONFIG_KEY",
        ),
        (
            lambda value: value["model_source"].update(
                initialization_seed=True
            ),
            "INVALID_INITIALIZATION_SEED",
        ),
    ],
)
def test_v2_source_missing_conflicting_unknown_and_bool_are_rejected(
    mutation, reason
):
    payload = _v2_payload()
    mutation(payload)
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == reason


def test_json_yaml_semantic_parity_and_relative_paths(tmp_path):
    payload = _v2_payload()
    json_path = tmp_path / "run.json"
    yaml_path = tmp_path / "run.yaml"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    json_config = load_training_run_config(json_path)
    yaml_config = load_training_run_config(yaml_path)
    assert json_config.to_dict() == yaml_config.to_dict()
    assert json_config.config_fingerprint == yaml_config.config_fingerprint
    assert json_config.model_source.reference_templates[0].poscar_path == (
        "references/NbC.POSCAR"
    )
    assert json_config.source_path == str(json_path.resolve())
    assert yaml_config.source_path == str(yaml_path.resolve())


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        (
            "schema_version: refsite_training_run_config_v2\n"
            "schema_version: refsite_training_run_config_v2\n",
            "CONFLICTING_CONFIG_KEY",
        ),
        ("value: !!python/object:builtins.object {}\n", "INVALID_CONFIG_YAML"),
        ("value: .nan\n", "NONFINITE_CONFIG_VALUE"),
        ("value: 2026-09-04\n", "INVALID_CONFIG_YAML_TYPE"),
    ],
)
def test_yaml_duplicate_object_tag_nonfinite_and_nonplain_rejected(
    tmp_path, contents, reason
):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(TrainingRunConfigError) as caught:
        load_training_run_config(path)
    assert caught.value.reason_code == reason


def test_effective_override_precedence_is_immutable_and_path_origin_is_explicit(
    tmp_path,
):
    config = TrainingRunConfig.from_dict(_payload())
    original = config.to_dict()
    effective = apply_training_run_overrides(
        config,
        TrainingRunConfigOverrides(
            device="cpu",
            dtype="float32",
            max_epochs=9,
            batch_size=3,
            validation_batch_size=2,
            learning_rate=3.0e-4,
            r_ot=4.5,
            r_mp=3.25,
            output_directory="cli-output",
        ),
        cli_cwd=tmp_path,
    )
    assert config.to_dict() == original
    assert effective.schema_version == TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2
    assert effective.runtime.dtype == "float32"
    assert effective.fit.max_epochs == 9
    assert effective.data.batch_size == 3
    assert effective.data.effective_validation_batch_size == 2
    assert effective.optimizer.learning_rate == 3.0e-4
    assert (effective.radii.r_ot, effective.radii.r_mp) == (4.5, 3.25)
    assert effective.output_directory == "cli-output"
    assert effective.output_directory_base == str(tmp_path.resolve())
    assert effective.config_fingerprint != config.config_fingerprint

    unchanged = apply_training_run_overrides(
        config, TrainingRunConfigOverrides()
    )
    assert unchanged is config
    assert unchanged.schema_version == TRAINING_RUN_CONFIG_SCHEMA_VERSION


def test_scratch_radius_overrides_update_the_effective_construction_contract():
    config = TrainingRunConfig.from_dict(_v2_payload())
    effective = apply_training_run_overrides(
        config,
        TrainingRunConfigOverrides(r_ot=4.5, r_mp=3.25),
    )
    source = effective.model_source
    assert source.potential.transport_support.cutoff == 4.5
    assert source.potential.feature.r_cut == 3.25
    assert source.potential.higher_body.cutoff == 3.25
    assert source.reference_templates[0].builder.graph_cutoff == 3.25
    assert source.reference_templates[0].builder.graph_skin == 0.5
    assert config.model_source.potential.transport_support.cutoff == 4.0
    assert config.model_source.potential.feature.r_cut == 3.0


def test_scratch_config_resolution_is_read_only_and_uses_config_path_base(
    tmp_path, monkeypatch
):
    payload = _v2_payload()
    payload["output_directory"] = "must-not-exist"
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_training_run_config(config_path)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("scratch execution dependency must not be called")

    module = __import__(
        "refsite_mlip.config.training_run", fromlist=["training_run"]
    )
    monkeypatch.setattr(module, "_preflight_device", forbidden)
    monkeypatch.setattr(module, "_resolve_existing_file", forbidden)
    monkeypatch.setattr(module, "load_reference_site_model_bundle", forbidden)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    resolved = resolve_training_run(config)
    report = resolved.to_dict()
    assert report["status"] == "scratch_config_ready"
    assert report["execution"] == {
        "implemented": False,
        "reason_code": "SCRATCH_EXECUTION_NOT_IMPLEMENTED",
    }
    assert report["runtime"]["paths"]["reference_poscars"] == [
        {
            "template_id": "nbc_rocksalt_222_v1",
            "path": str(tmp_path / "references" / "NbC.POSCAR"),
        }
    ]
    assert not (tmp_path / "must-not-exist").exists()
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    observed_numpy_rng = np.random.get_state()
    assert observed_numpy_rng[0] == numpy_rng[0]
    assert np.array_equal(observed_numpy_rng[1], numpy_rng[1])
    assert observed_numpy_rng[2:] == numpy_rng[2:]


def test_v1_rejects_validation_batch_size_without_fingerprint_hiding():
    payload = _payload()
    payload["data"]["validation_batch_size"] = 1
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "UNKNOWN_CONFIG_KEY"
    assert caught.value.field == "data.validation_batch_size"


def test_v2_omitted_validation_batch_size_round_trip_keeps_dynamic_default():
    payload = _v2_payload(kind="bundle")
    payload["data"].pop("validation_batch_size")
    config = TrainingRunConfig.from_dict(payload)
    assert config.data.validation_batch_size is None
    assert "validation_batch_size" not in config.to_dict()["data"]
    round_trip = TrainingRunConfig.from_dict(config.to_dict())
    assert round_trip == config
    effective = apply_training_run_overrides(
        round_trip, TrainingRunConfigOverrides(batch_size=9)
    )
    assert effective.data.validation_batch_size is None
    assert effective.data.effective_validation_batch_size == 9


def test_optional_evaluation_policy_may_be_omitted_and_canonicalizes_to_null():
    payload = _v2_payload()
    template = payload["model_source"]["reference_templates"][0]
    template.pop("evaluation_policy")
    config = TrainingRunConfig.from_dict(payload)
    assert config.model_source.reference_templates[0].evaluation_policy is None
    assert (
        config.to_dict()["model_source"]["reference_templates"][0][
            "evaluation_policy"
        ]
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["model_source"]["potential"]["feature"].pop(
            "e3nn_normalization"
        ),
        lambda value: value["model_source"]["potential"]["higher_body"].update(
            lmax=True
        ),
        lambda value: value["model_source"]["reference_templates"][0][
            "builder"
        ].update(maximum_strain=False),
        lambda value: value["model_source"]["reference_templates"][0][
            "phase_specification"
        ].update(mode_weights=[True, 1.0, 1.0]),
        lambda value: value["model_source"]["reference_templates"][0].update(
            poscar_path=123
        ),
    ],
)
def test_scratch_nested_missing_bool_and_path_coercions_are_rejected(mutation):
    payload = _v2_payload()
    mutation(payload)
    with pytest.raises(TrainingRunConfigError):
        TrainingRunConfig.from_dict(payload)


def test_scratch_numeric_and_mapping_order_canonicalization_stabilizes_sha():
    first = _v2_payload()
    second = json.loads(json.dumps(first))
    second["model_source"]["potential"]["feature"]["r_cut"] = 3
    second["model_source"]["potential"]["higher_body"]["cutoff"] = 3
    second["model_source"]["reference_templates"][0]["builder"][
        "graph_cutoff"
    ] = 3
    second = dict(reversed(tuple(second.items())))
    first_config = TrainingRunConfig.from_dict(first)
    second_config = TrainingRunConfig.from_dict(second)
    assert first_config.to_dict() == second_config.to_dict()
    assert first_config.config_fingerprint == second_config.config_fingerprint


def test_json_suffix_compatibility_yml_and_invalid_schema_type(tmp_path):
    payload = _payload()
    extensionless = tmp_path / "legacy-config"
    extensionless.write_text(json.dumps(payload), encoding="utf-8")
    assert load_training_run_config(extensionless).schema_version == (
        TRAINING_RUN_CONFIG_SCHEMA_VERSION
    )

    yml_path = tmp_path / "run.yml"
    yml_path.write_text(yaml.safe_dump(_v2_payload()), encoding="utf-8")
    assert load_training_run_config(yml_path).schema_version == (
        TRAINING_RUN_CONFIG_SCHEMA_VERSION_V2
    )

    payload["schema_version"] = []
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "UNSUPPORTED_TRAINING_RUN_SCHEMA"


def test_scratch_radius_mismatch_is_actionable():
    payload = _v2_payload()
    payload["radii"]["r_mp"] = 3.25
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "RADIUS_MODEL_MISMATCH"
    assert caught.value.expected == 3.25
    assert caught.value.actual == 3.0


def test_scratch_template_order_identity_and_default_are_strict():
    payload = _v2_payload()
    template = payload["model_source"]["reference_templates"][0]
    payload["model_source"]["reference_templates"].append(
        json.loads(json.dumps(template))
    )
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "DUPLICATE_TEMPLATE_ID"

    payload = _v2_payload()
    payload["model_source"]["default_template_id"] = "unknown-template"
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "MISSING_DEFAULT_TEMPLATE"

    payload = _v2_payload()
    payload["model_source"]["reference_templates"][0]["builder"][
        "strict_domain"
    ]["supercell_shape"] = [2, True, 2]
    with pytest.raises(TrainingRunConfigError) as caught:
        TrainingRunConfig.from_dict(payload)
    assert caught.value.reason_code == "INVALID_MODEL_SOURCE_CONFIG"
