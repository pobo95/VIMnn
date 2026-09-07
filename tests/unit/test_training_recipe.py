from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import random

import numpy as np
import pytest
import torch
import yaml

from refsite_mlip.config import (
    ReferenceSpecificationConfig,
    RecipeModelConfig,
    TrainingRecipeConfig,
    TrainingRecipeError,
    TrainingRunConfig,
    TrainingRunConfigOverrides,
    compile_training_recipe,
    load_training_recipe,
)
from refsite_mlip.data import (
    PhaseSpecification,
    ReferenceTemplateBuilderConfig,
    StrictTemplateDomain,
)
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import EvaluationPolicy


MINIMAL_RECIPE_YAML = """\
schema_version: refsite_training_recipe_v1
name: si-run

model:
  num_interactions: 2
  correlation: 3
  hidden_channels: 8
  max_L: 2

reference:
  specification: reference-specification.yaml
  poscar: POSCAR
  allow_provisional_phase: true

data:
  train: train.xyz
  validation: validate.xyz

ot_solver:
  training: sinkhorn
  inference: sinkhorn_newton_krylov

loss:
  energy_weight: 1.0
  forces_weight: 100.0
  stress_weight: 0.0

baseline: minimum_norm

training:
  batch_size: 4
  validation_batch_size: 8
  max_epochs: 100
  learning_rate: 1.0e-3

runtime:
  device: cpu
  dtype: float64
  seed: 17
"""


def _builder() -> ReferenceTemplateBuilderConfig:
    return ReferenceTemplateBuilderConfig(
        template_id="silicon-222",
        strict_domain=StrictTemplateDomain(
            reference_site_count=2,
            supercell_shape=(1, 1, 1),
            species_vocabulary=(6, 14),
            reference_composition=(1, 1),
            allowed_compositions=((1, 1), (0, 1)),
            allowed_num_atoms=(2, 1),
            allowed_vacancy_masses=(0, 1),
        ),
        site_type_ids=(0, 1),
        graph_cutoff=3.0,
        graph_skin=0.5,
        maximum_strain=0.1,
        expected_stabilizer_size=1,
    )


def _phase(*, approval="provisional") -> PhaseSpecification:
    return PhaseSpecification(
        modes=torch.eye(3, dtype=torch.long),
        mode_weights=torch.ones(3, dtype=torch.float64),
        site_type_alignment_weights=torch.eye(2, dtype=torch.float64),
        channel_weights=torch.ones(2, dtype=torch.float64),
        approval_status=approval,
    )


def _policy(*, template_id="silicon-222", template_fingerprint="b" * 64):
    return EvaluationPolicy(
        template_id=template_id,
        template_fingerprint=template_fingerprint,
        candidate_offsets=torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=torch.float64
        ),
        phase_step_schedule=(0.5,),
        phase_damping_schedule=(1.0,),
        minimum_objective_gap_absolute=1.0e-8,
        minimum_cross_amplitude_absolute=1.0e-8,
        minimum_atomic_amplitude_absolute=1.0e-8,
        minimum_reference_amplitude_absolute=1.0e-8,
        minimum_curvature=1.0e-8,
        maximum_condition=1.0e6,
        maximum_gradient_norm=1.0e-6,
        equivalence_tolerance=1.0e-10,
    )


def _spec(*, approval="provisional", policy=False) -> ReferenceSpecificationConfig:
    return ReferenceSpecificationConfig(
        builder=_builder(),
        phase_specification=_phase(approval=approval),
        evaluation_policy=_policy() if policy else None,
        species_alignment_weights=((1.0, -0.5), (-1.0, 2.0)),
        poscar_sha256="a" * 64,
    )


def _payload(*, alias=False, approval=True):
    model = {
        "num_interactions": 2,
        "correlation": 3,
        "initialization_seed": 31,
    }
    if alias:
        model["hidden_irreps"] = "8x0e+ 8x1o +8x2e"
    else:
        model.update(hidden_channels=8, max_L=2)
    return {
        "schema_version": "refsite_training_recipe_v1",
        "name": "si-smoke",
        "model": model,
        "reference": {
            "specification": "reference.yaml",
            "poscar": "POSCAR",
            "allow_provisional_phase": approval,
        },
        "data": {"train": "train.xyz", "validation": "validate.xyz"},
        "ot_solver": {"training": "sinkhorn", "inference": "sinkhorn"},
        "loss": {
            "energy_weight": 1.0,
            "forces_weight": 100.0,
            "stress_weight": 10.0,
        },
        "baseline": "minimum_norm",
        "radii": {"r_ot": 4.0, "r_mp": 3.0},
        "training": {
            "batch_size": 2,
            "validation_batch_size": 3,
            "max_epochs": 4,
            "learning_rate": 0.0005,
        },
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 17},
    }


def test_documented_39_line_minimal_recipe_uses_symmetric_defaults():
    assert len(MINIMAL_RECIPE_YAML.splitlines()) == 39
    recipe = TrainingRecipeConfig.from_dict(yaml.safe_load(MINIMAL_RECIPE_YAML))
    resolved = compile_training_recipe(recipe, (_spec(policy=True),))
    higher = resolved.config.model_source.potential.higher_body
    assert recipe.model.correlation_method == "symmetric"
    assert higher.contract_version == "central_conditioned_symmetric_power_v2"
    assert higher.symmetric_correlation.correlation_order == 3
    assert resolved.config.data.batch_size == 4
    assert resolved.config.data.effective_validation_batch_size == 8
    assert resolved.manifest.inference_ot_solver == "sinkhorn_newton_krylov"


def test_minimal_recipe_compiles_to_complete_canonical_v2_deterministically():
    first_recipe = TrainingRecipeConfig.from_dict(_payload())
    second_recipe = TrainingRecipeConfig.from_dict(dict(reversed(tuple(_payload().items()))))
    first = compile_training_recipe(first_recipe, (_spec(),))
    second = compile_training_recipe(second_recipe, (_spec(),))

    assert first.config.schema_version == "refsite_training_run_config_v2"
    assert first.config.canonical_json() == second.config.canonical_json()
    assert first.config.config_fingerprint == second.config.config_fingerprint
    assert first.manifest.content_fingerprint == second.manifest.content_fingerprint
    assert TrainingRunConfig.from_json(first.config.canonical_json()).to_dict() == first.config.to_dict()

    config = first.config
    higher = config.model_source.potential.higher_body
    assert config.model_source.potential.num_layers == 2
    assert higher.contract_version == "central_conditioned_symmetric_power_v2"
    assert higher.n_correlation_channels == 8
    assert higher.lmax == config.model_source.potential.feature.lmax == 2
    assert higher.symmetric_correlation.correlation_order == 3
    assert higher.symmetric_correlation.basis_kind == "full_path"
    assert higher.symmetric_correlation.normalization == "component"
    assert config.data.batch_size == 2
    assert config.data.effective_validation_batch_size == 3
    assert config.loss.force_weight == 100.0
    assert config.optimizer.learning_rate == 0.0005
    assert config.fit.max_epochs == 4
    assert config.output_directory == "runs/si-smoke"
    assert config.train_step.solver_path == config.validation_step.solver_path == "train_fixed"
    assert config.baseline.rank_policy == "minimum_norm"
    assert config.radii.derived.to_dict() == {
        "length_unit": "angstrom",
        "r_on_ot": 3.5,
        "r_off_ot": 4.0,
        "r_candidate_ot": 4.2,
        "r_mp": 3.0,
        "r_candidate_mp": 3.5,
    }


def test_hidden_irreps_alias_and_explicit_layout_compile_identically():
    explicit = compile_training_recipe(TrainingRecipeConfig.from_dict(_payload()), (_spec(),))
    alias = compile_training_recipe(TrainingRecipeConfig.from_dict(_payload(alias=True)), (_spec(),))
    assert explicit.config.canonical_json() == alias.config.canonical_json()
    assert explicit.config.config_fingerprint == alias.config.config_fingerprint

    for invalid in (
        "8x0e + 4x1o + 8x2e",
        "8x0e + 8x2e",
        "8x0e + 8x1e",
        "8x0e + 8x1o + 8x2e + 8x3o",
        "8x0e + 8x0e",
    ):
        with pytest.raises(TrainingRecipeError):
            RecipeModelConfig(hidden_irreps=invalid)
    with pytest.raises(TrainingRecipeError, match="conflicts"):
        TrainingRecipeConfig.from_dict(
            {**_payload(alias=True), "model": {**_payload(alias=True)["model"], "max_L": 2}}
        )


@pytest.mark.parametrize("channels", [17, 64])
def test_symmetric_recipe_compiles_large_hidden_channel_counts(channels):
    payload = _payload()
    payload["model"]["hidden_channels"] = channels
    resolved = compile_training_recipe(
        TrainingRecipeConfig.from_dict(payload), (_spec(),)
    )
    higher = resolved.config.model_source.potential.higher_body
    assert higher.n_correlation_channels == channels
    higher.validate()
    restored = HigherBodyConfig.from_dict(higher.to_dict())
    assert restored == higher
    assert restored.canonical_json() == higher.canonical_json()


def test_large_hidden_irreps_alias_canonicalizes_to_uniform_channels():
    payload = _payload(alias=True)
    payload["model"]["hidden_irreps"] = "64x0e + 64x1o + 64x2e"
    resolved = compile_training_recipe(
        TrainingRecipeConfig.from_dict(payload), (_spec(),)
    )
    higher = resolved.config.model_source.potential.higher_body
    assert resolved.recipe.model.hidden_irreps == RecipeModelConfig(
        hidden_irreps="64x0e+64x1o+64x2e"
    ).hidden_irreps
    assert higher.n_correlation_channels == 64
    assert higher.lmax == 2


def test_omitted_and_explicit_symmetric_method_are_canonically_identical():
    implicit_payload = _payload()
    implicit_payload["model"].pop("correlation")
    explicit_payload = _payload()
    explicit_payload["model"]["correlation_method"] = "symmetric"

    implicit = compile_training_recipe(
        TrainingRecipeConfig.from_dict(implicit_payload), (_spec(),)
    )
    explicit = compile_training_recipe(
        TrainingRecipeConfig.from_dict(explicit_payload), (_spec(),)
    )

    assert implicit.config.to_dict() == explicit.config.to_dict()
    assert implicit.config.canonical_json() == explicit.config.canonical_json()
    assert implicit.config.config_fingerprint == explicit.config.config_fingerprint
    assert (
        implicit.config.model_source.potential.higher_body.content_fingerprint
        == explicit.config.model_source.potential.higher_body.content_fingerprint
    )
    assert implicit.recipe.model.correlation_method == "symmetric"


def test_sequential_recipe_matches_hand_written_legacy_canonical_config():
    symmetric = compile_training_recipe(
        TrainingRecipeConfig.from_dict(_payload()), (_spec(),)
    )
    payload = _payload()
    payload["model"] = {
        "correlation_method": "sequential",
        "correlation_mode": "uuu",
        "num_interactions": 2,
        "hidden_channels": 8,
        "max_L": 2,
        "initialization_seed": 31,
    }
    sequential = compile_training_recipe(
        TrainingRecipeConfig.from_dict(payload), (_spec(),)
    )

    symmetric_source = symmetric.config.model_source
    symmetric_potential = symmetric_source.potential
    previous = symmetric_potential.higher_body
    hand_written_higher = HigherBodyConfig(
        irreps_feature=previous.irreps_feature,
        species_count=previous.species_count,
        site_type_count=previous.site_type_count,
        site_type_embedding_dim=previous.site_type_embedding_dim,
        n_correlation_channels=previous.n_correlation_channels,
        lmax=2,
        radial_feature_dim=previous.radial_feature_dim,
        radial_hidden_dims=previous.radial_hidden_dims,
        avg_num_neighbors=previous.avg_num_neighbors,
        cutoff=previous.cutoff,
        edge_length_scale=previous.edge_length_scale,
        correlation_mode="uuu",
    )
    expected = replace(
        symmetric.config,
        model_source=replace(
            symmetric_source,
            potential=replace(
                symmetric_potential,
                higher_body=hand_written_higher,
            ),
        ),
    )
    assert sequential.config.to_dict() == expected.to_dict()
    assert sequential.config.canonical_json() == expected.canonical_json()
    assert sequential.config.config_fingerprint == expected.config_fingerprint
    assert sequential.config.model_source.potential.higher_body.symmetric_correlation is None

    uvw_payload = _payload()
    uvw_payload["model"] = {
        "correlation_method": "sequential",
        "correlation_mode": "uvw",
        "num_interactions": 2,
        "hidden_channels": 2,
        "max_L": 2,
    }
    uvw = compile_training_recipe(
        TrainingRecipeConfig.from_dict(uvw_payload), (_spec(),)
    )
    assert uvw.config.model_source.potential.higher_body.correlation_mode == "uvw"


def test_recipe_model_vocabulary_rejects_internal_and_conflicting_names():
    for forbidden in (
        "v1",
        "v2",
        "symmetric_power_v2",
        "central_conditioned_symmetric_power_v2",
        "central_conditioned_higher_body_v1",
    ):
        payload = _payload()
        payload["model"]["correlation_method"] = forbidden
        with pytest.raises(TrainingRecipeError) as caught:
            TrainingRecipeConfig.from_dict(payload)
        assert caught.value.reason_code == "UNSUPPORTED_CORRELATION_METHOD"

    for forbidden_key in ("architecture", "contract_version", "basis_kind"):
        payload = _payload()
        payload["model"][forbidden_key] = "internal"
        with pytest.raises(TrainingRecipeError) as caught:
            TrainingRecipeConfig.from_dict(payload)
        assert caught.value.reason_code == "UNKNOWN_RECIPE_KEY"

    sequential = _payload()
    sequential["model"].update(
        correlation_method="sequential", correlation_mode="uuu"
    )
    with pytest.raises(TrainingRecipeError, match="forbids correlation"):
        TrainingRecipeConfig.from_dict(sequential)

    symmetric = _payload()
    symmetric["model"]["correlation_mode"] = "uuu"
    with pytest.raises(TrainingRecipeError, match="only for sequential"):
        TrainingRecipeConfig.from_dict(symmetric)

    for update, reason in (
        ({"max_L": 1}, "UNSUPPORTED_SEQUENTIAL_MAX_L"),
        ({"hidden_irreps": "8x0e + 8x1o + 8x2e"}, "UNSUPPORTED_SEQUENTIAL_HIDDEN_IRREPS"),
    ):
        payload = _payload()
        payload["model"] = {
            "correlation_method": "sequential",
            "correlation_mode": "uuu",
            "num_interactions": 2,
            "hidden_channels": 8,
            "max_L": 2,
            **update,
        }
        if "hidden_irreps" in update:
            payload["model"].pop("hidden_channels")
            payload["model"].pop("max_L")
        with pytest.raises(TrainingRecipeError) as caught:
            TrainingRecipeConfig.from_dict(payload)
        assert caught.value.reason_code == reason


def test_public_ot_solver_tokens_and_inference_preference_are_strict():
    policy_spec = _spec(policy=True)
    fixed_payload = _payload()
    adaptive_payload = _payload()
    adaptive_payload["ot_solver"]["inference"] = "sinkhorn_newton_krylov"
    fixed = compile_training_recipe(
        TrainingRecipeConfig.from_dict(fixed_payload), (policy_spec,)
    )
    adaptive = compile_training_recipe(
        TrainingRecipeConfig.from_dict(adaptive_payload), (policy_spec,)
    )
    assert fixed.config.canonical_json() == adaptive.config.canonical_json()
    assert fixed.config.config_fingerprint == adaptive.config.config_fingerprint
    assert fixed.manifest.inference_ot_solver == "sinkhorn"
    assert adaptive.manifest.inference_ot_solver == "sinkhorn_newton_krylov"
    assert fixed.manifest.content_fingerprint != adaptive.manifest.content_fingerprint

    missing_policy = _payload()
    missing_policy["ot_solver"]["inference"] = "sinkhorn_newton_krylov"
    with pytest.raises(TrainingRecipeError) as caught:
        compile_training_recipe(
            TrainingRecipeConfig.from_dict(missing_policy), (_spec(),)
        )
    assert caught.value.reason_code == "INFERENCE_EVALUATION_POLICY_REQUIRED"

    training_adaptive = _payload()
    training_adaptive["ot_solver"]["training"] = "sinkhorn_newton_krylov"
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(training_adaptive)
    assert caught.value.reason_code == "UNSUPPORTED_TRAINING_OT_SOLVER"

    for token in ("fixed_sinkhorn", "train_fixed", "adaptive_sinkhorn"):
        payload = _payload()
        payload["ot_solver"]["training"] = token
        with pytest.raises(TrainingRecipeError) as caught:
            TrainingRecipeConfig.from_dict(payload)
        assert caught.value.reason_code == "UNSUPPORTED_TRAINING_OT_SOLVER"

    for token in (
        "fixed_sinkhorn",
        "adaptive_sinkhorn",
        "hybrid_sinkhorn_newton_krylov",
        "train_fixed",
        "eval_adaptive",
        "sinkhorn-newton-krylov",
    ):
        payload = _payload()
        payload["ot_solver"]["inference"] = token
        with pytest.raises(TrainingRecipeError) as caught:
            TrainingRecipeConfig.from_dict(payload)
        assert caught.value.reason_code == "UNSUPPORTED_INFERENCE_OT_SOLVER"


def test_training_and_validation_batch_sizes_map_and_override_independently():
    recipe = TrainingRecipeConfig.from_dict(_payload())
    resolved = compile_training_recipe(recipe, (_spec(),))
    assert resolved.config.data.batch_size == 2
    assert resolved.config.data.effective_validation_batch_size == 3
    origins = dict(resolved.manifest.field_origins)
    assert origins["data.batch_size"] == "user"
    assert origins["data.validation_batch_size"] == "user"

    omitted_payload = _payload()
    omitted_payload["training"].pop("validation_batch_size")
    omitted = compile_training_recipe(
        TrainingRecipeConfig.from_dict(omitted_payload), (_spec(),)
    )
    assert omitted.config.data.batch_size == 2
    assert omitted.config.data.effective_validation_batch_size == 2
    assert dict(omitted.manifest.field_origins)["data.validation_batch_size"] == "preset"

    overridden = compile_training_recipe(
        recipe,
        (_spec(),),
        overrides=TrainingRunConfigOverrides(
            batch_size=5,
            validation_batch_size=7,
        ),
    )
    assert overridden.config.data.batch_size == 5
    assert overridden.config.data.effective_validation_batch_size == 7
    origins = dict(overridden.manifest.field_origins)
    assert origins["data.batch_size"] == "CLI"
    assert origins["data.validation_batch_size"] == "CLI"
    assert overridden.config.config_fingerprint != resolved.config.config_fingerprint

    for field_name, invalid in (
        ("batch_size", True),
        ("validation_batch_size", True),
        ("validation_batch_size", 0),
        ("validation_batch_size", -1),
        ("validation_batch_size", 1.5),
    ):
        payload = _payload()
        payload["training"][field_name] = invalid
        with pytest.raises(TrainingRecipeError):
            TrainingRecipeConfig.from_dict(payload)


def test_recipe_early_stopping_patience_maps_to_validation_selection():
    default_recipe = TrainingRecipeConfig.from_dict(_payload())
    default_resolved = compile_training_recipe(default_recipe, (_spec(),))
    assert default_recipe.training.early_stopping_patience == 15
    assert default_recipe.to_dict()["training"]["early_stopping_patience"] == 15
    assert default_resolved.config.selection.early_stopping_patience == 15
    assert dict(default_resolved.manifest.field_origins)[
        "selection.early_stopping_patience"
    ] == "preset"
    assert dict(default_resolved.manifest.preset_versions)["training"] == (
        "training_defaults_v2"
    )

    payload = _payload()
    payload["training"]["early_stopping_patience"] = 15
    recipe = TrainingRecipeConfig.from_dict(payload)
    resolved = compile_training_recipe(recipe, (_spec(),))
    assert recipe.training.early_stopping_patience == 15
    assert recipe.to_dict()["training"]["early_stopping_patience"] == 15
    assert resolved.config.selection.monitor == "total_loss"
    assert resolved.config.selection.mode == "min"
    assert resolved.config.selection.min_delta == 0.0
    assert resolved.config.selection.early_stopping_patience == 15
    assert dict(resolved.manifest.field_origins)[
        "selection.early_stopping_patience"
    ] == "user"
    assert resolved.config.config_fingerprint == (
        default_resolved.config.config_fingerprint
    )

    disabled_payload = _payload()
    disabled_payload["training"]["early_stopping_patience"] = None
    disabled_recipe = TrainingRecipeConfig.from_dict(disabled_payload)
    disabled_resolved = compile_training_recipe(disabled_recipe, (_spec(),))
    assert disabled_recipe.to_dict()["training"]["early_stopping_patience"] is None
    assert disabled_resolved.config.selection.early_stopping_patience is None
    assert dict(disabled_resolved.manifest.field_origins)[
        "selection.early_stopping_patience"
    ] == "user"
    assert disabled_resolved.config.config_fingerprint != (
        default_resolved.config.config_fingerprint
    )


@pytest.mark.parametrize("invalid", (True, -1, 1.5, "15"))
def test_recipe_early_stopping_patience_rejects_invalid_values(invalid):
    payload = _payload()
    payload["training"]["early_stopping_patience"] = invalid
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(payload)
    assert caught.value.reason_code == "INVALID_RECIPE_INTEGER"
    assert caught.value.field == "training.early_stopping_patience"


@pytest.mark.parametrize("maximum_l", (0, 1, 2))
def test_supported_max_l_maps_to_feature_higher_body_and_natural_irreps(maximum_l):
    payload = _payload()
    payload["model"]["max_L"] = maximum_l
    resolved = compile_training_recipe(TrainingRecipeConfig.from_dict(payload), (_spec(),))
    potential = resolved.config.model_source.potential
    assert potential.feature.lmax == potential.higher_body.lmax == maximum_l
    expected = ["2x0e"] + [
        f"6x{angular}{'e' if angular % 2 == 0 else 'o'}"
        for angular in range(maximum_l + 1)
    ]
    assert potential.higher_body.irreps_feature == "+".join(expected)
    assert potential.higher_body.n_correlation_channels == 8


def test_provisional_phase_and_reference_radius_contracts_are_explicit():
    with pytest.raises(TrainingRecipeError) as caught:
        compile_training_recipe(TrainingRecipeConfig.from_dict(_payload(approval=False)), (_spec(),))
    assert caught.value.reason_code == "PROVISIONAL_PHASE_NOT_APPROVED"

    wrong = ReferenceSpecificationConfig(
        builder=ReferenceTemplateBuilderConfig.from_dict(
            {**_builder().to_dict(), "graph_cutoff": 2.5}
        ),
        phase_specification=_phase(approval="production_approved"),
        evaluation_policy=None,
        species_alignment_weights=((1.0, -0.5), (-1.0, 2.0)),
        poscar_sha256="a" * 64,
    )
    with pytest.raises(TrainingRecipeError) as caught:
        compile_training_recipe(TrainingRecipeConfig.from_dict(_payload()), (wrong,))
    assert caught.value.reason_code == "REFERENCE_RADIUS_MISMATCH"


def test_baseline_policy_is_never_silently_changed():
    expected = {"zero": None, "fit_full_rank": "error", "minimum_norm": "minimum_norm"}
    for policy, rank in expected.items():
        payload = _payload()
        payload["baseline"] = policy
        config = compile_training_recipe(TrainingRecipeConfig.from_dict(payload), (_spec(),)).config
        assert (None if config.baseline is None else config.baseline.rank_policy) == rank


def test_recipe_validation_is_strict_frozen_and_rng_neutral():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    recipe = TrainingRecipeConfig.from_dict(_payload())
    compile_training_recipe(recipe, (_spec(),))
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(FrozenInstanceError):
        recipe.name = "changed"

    for field_name, value in (
        ("num_interactions", True),
        ("correlation", 4),
        ("hidden_channels", 0),
        ("max_L", 3),
    ):
        payload = _payload()
        payload["model"][field_name] = value
        with pytest.raises(TrainingRecipeError):
            TrainingRecipeConfig.from_dict(payload)
    payload = _payload()
    payload["loss"]["huber"] = True
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(payload)
    assert caught.value.reason_code == "UNKNOWN_RECIPE_KEY"


@pytest.mark.parametrize("value", [-1, True, 64.0, "64"])
def test_recipe_hidden_channels_remains_a_strict_positive_integer(value):
    payload = _payload()
    payload["model"]["hidden_channels"] = value
    with pytest.raises(TrainingRecipeError) as caught:
        TrainingRecipeConfig.from_dict(payload)
    assert caught.value.reason_code == "INVALID_RECIPE_INTEGER"


def test_reference_specification_fingerprint_detects_semantic_corruption():
    spec = _spec()
    payload = spec.to_dict()
    assert payload["content_fingerprint"] == spec.content_fingerprint
    assert ReferenceSpecificationConfig.from_dict(payload).to_dict() == payload
    corrupted = json.loads(json.dumps(payload))
    corrupted["poscar_sha256"] = "b" * 64
    with pytest.raises(TrainingRecipeError) as caught:
        ReferenceSpecificationConfig.from_dict(corrupted)
    assert caught.value.reason_code == "REFERENCE_SPECIFICATION_FINGERPRINT_MISMATCH"


def test_multi_template_data_requires_explicit_assignment():
    payload = _payload()
    payload["reference"] = {
        "templates": [
            {"specification": "a.yaml", "poscar": "A.POSCAR", "allow_provisional_phase": True},
            {"specification": "b.yaml", "poscar": "B.POSCAR", "allow_provisional_phase": True},
        ],
        "default_template_id": "silicon-222",
    }
    second = ReferenceSpecificationConfig(
        builder=replace(_builder(), template_id="silicon-alt"),
        phase_specification=_phase(), evaluation_policy=None,
        species_alignment_weights=((1.0, -0.5), (-1.0, 2.0)),
        poscar_sha256="b" * 64,
    )
    with pytest.raises(TrainingRecipeError) as caught:
        compile_training_recipe(
            TrainingRecipeConfig.from_dict(payload), (_spec(), second)
        )
    assert caught.value.reason_code == "MISSING_TEMPLATE_ASSIGNMENT"

    payload["data"] = {
        "train": [{"path": "train.xyz", "template_id": "silicon-222"}],
        "validation": [
            {"path": "validate.xyz", "template_id": "silicon-alt"}
        ],
    }
    config = compile_training_recipe(
        TrainingRecipeConfig.from_dict(payload), (_spec(), second)
    ).config
    assert config.data.train[0].template_id == "silicon-222"
    assert config.data.validation[0].template_id == "silicon-alt"


def test_canonical_run_config_existing_bytes_are_unchanged_by_recipe_module():
    compiled = compile_training_recipe(TrainingRecipeConfig.from_dict(_payload()), (_spec(),)).config
    encoded = compiled.canonical_json()
    assert hashlib.sha256(encoded.encode()).hexdigest() == compiled.config_fingerprint
    assert "recipe" not in compiled.to_dict()
    assert "resolution_manifest" not in compiled.to_dict()


def test_json_yaml_semantic_parity(tmp_path):
    payload = _payload()
    json_path = tmp_path / "recipe.json"
    yaml_path = tmp_path / "recipe.yaml"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    from_json = load_training_recipe(json_path)
    from_yaml = load_training_recipe(yaml_path)
    assert from_json.to_dict() == from_yaml.to_dict()
    assert from_json.content_fingerprint == from_yaml.content_fingerprint


def test_input_toctou_is_detected(monkeypatch, tmp_path):
    import refsite_mlip.config.training_recipe as module

    path = tmp_path / "input.bin"
    path.write_bytes(b"initial")
    original_read = module.os.read
    changed = False

    def racing_read(descriptor, count):
        nonlocal changed
        value = original_read(descriptor, count)
        if value and not changed:
            changed = True
            with path.open("ab") as stream:
                stream.write(b"-changed")
                stream.flush()
                module.os.fsync(stream.fileno())
        return value

    monkeypatch.setattr(module.os, "read", racing_read)
    with pytest.raises(TrainingRecipeError) as caught:
        module._read_regular_file(path, text=False)
    assert caught.value.reason_code == "INPUT_TOCTOU_MISMATCH"
