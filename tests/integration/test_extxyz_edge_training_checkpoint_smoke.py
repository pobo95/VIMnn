from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from refsite_mlip.data import (
    ExtXYZLoadConfig,
    ReferenceTemplate,
    StrictTemplateDomain,
    TemplateRegistry,
    collate_structure_samples,
    load_extxyz_samples,
)
from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.graph import build_reference_graph_topology
from refsite_mlip.interactions import HigherBodyConfig
from refsite_mlip.models import (
    PotentialConfig,
    ReferenceSitePotential,
    TemplateExecutionContext,
    evaluate_structure_batch,
)
from refsite_mlip.phase import find_typed_stabilizer
from refsite_mlip.training import (
    FitConfig,
    FitProgress,
    LossConfig,
    ModelSelectionConfig,
    ModelSelectionState,
    OptimizerConfig,
    ResumePolicy,
    SchedulerConfig,
    TrainStepConfig,
    ValidationStepConfig,
    build_optimizer,
    build_scheduler,
    capture_training_checkpoint,
    compute_potential_loss,
    fingerprint_batch_sequence,
    load_training_checkpoint,
    restore_training_checkpoint_,
    save_training_checkpoint,
    train_step,
)
from refsite_mlip.transport import TransportSupportConfig


TEMPLATE_ID = "synthetic_extxyz_edge_pipeline"


def _registry() -> TemplateRegistry:
    fractional = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64
    )
    site_types = torch.tensor([0, 1], dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64) * 4.0
    topology = build_reference_graph_topology(
        fractional,
        site_types,
        cell,
        cutoff=3.0,
        skin=0.5,
        maximum_strain=0.1,
    )
    domain = StrictTemplateDomain(
        reference_site_count=2,
        supercell_shape=(1, 1, 1),
        species_vocabulary=(6, 41),
        reference_composition=(1, 1),
        allowed_compositions=((1, 1), (0, 1)),
        allowed_num_atoms=(2, 1),
        allowed_vacancy_masses=(0, 1),
    )
    template = ReferenceTemplate.snapshot(
        TEMPLATE_ID,
        topology,
        torch.eye(3, dtype=torch.long),
        torch.ones(3, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        find_typed_stabilizer(fractional, site_types),
        (6, 41),
        strict_domain=domain,
    )
    registry = TemplateRegistry()
    registry.add(template)
    return registry


def _frame(*, vacancy: bool) -> Atoms:
    numbers = [41] if vacancy else [6, 41]
    positions = (
        np.array([[2.0, 2.0, 2.0]])
        if vacancy
        else np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    )
    atoms = Atoms(numbers=numbers, positions=positions, cell=np.eye(3) * 4.0, pbc=True)
    forces = (
        np.array([[0.04, -0.03, 0.02]])
        if vacancy
        else np.array([[0.02, -0.01, 0.03], [-0.02, 0.01, -0.03]])
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.25 if vacancy else -2.5,
        forces=forces,
        stress=np.array([0.01, 0.02, 0.03, 0.004, -0.005, 0.006]),
    )
    return atoms


def _model(template: ReferenceTemplate) -> ReferenceSitePotential:
    torch.manual_seed(17)
    feature = ProbabilityMultipoleConfig(
        species_vocabulary=(6, 41),
        n_radial=1,
        lmax=2,
        ell_feature=1.0,
        r_cut=3.0,
        probability_tolerance=None,
        site_type_vocabulary=(0, 1),
    )
    higher = HigherBodyConfig(
        irreps_feature="2x0e+2x0e+2x1o+2x2e",
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
    config = PotentialConfig(
        species_vocabulary=(6, 41),
        num_layers=1,
        feature=feature,
        higher_body=higher,
        readout_hidden=8,
        energy_scale=1.0,
        transport_support=TransportSupportConfig(
            kind="compact_c2",
            cutoff=4.0,
            switch_width=0.5,
            candidate_skin=0.2,
            backend="edge_list",
        ),
    )
    return ReferenceSitePotential(
        config,
        template.topology,
        template.phase_modes,
        template.phase_mode_weights,
        torch.eye(2, dtype=torch.float64),
        template.site_alignment_weights,
        template.phase_channel_weights,
        (0.0, 0.0),
    ).to(dtype=torch.float64)


def test_synthetic_extxyz_edge_training_and_checkpoint_restore(tmp_path):
    registry = _registry()
    source = tmp_path / "synthetic.xyz"
    write(source, [_frame(vacancy=False), _frame(vacancy=True)], format="extxyz")
    loaded = load_extxyz_samples(
        ExtXYZLoadConfig(
            source_path=str(source),
            sample_id_prefix="synthetic",
            template_id=TEMPLATE_ID,
        ),
        registry,
    )
    assert tuple(sample.sample_id for sample in loaded.samples) == (
        "synthetic:000000",
        "synthetic:000001",
    )
    batch = collate_structure_samples(loaded.samples, registry)
    template = registry.resolve(TEMPLATE_ID)
    contexts = {
        TEMPLATE_ID: TemplateExecutionContext.from_reference_template(
            template, avg_num_neighbors=6.0
        )
    }
    model = _model(template)
    baseline = model.atomic_baseline.clone()
    input_fingerprint = fingerprint_batch_sequence((batch,), split_name="synthetic")
    loss_config = LossConfig(
        energy_weight=1.0,
        force_weight=1.0,
        stress_weight=0.01,
        energy_normalization="per_atom",
        stress_scale=0.1,
    )

    differentiable_batch = replace(
        batch,
        positions=batch.positions.detach().clone().requires_grad_(True),
    )
    output = evaluate_structure_batch(
        model,
        differentiable_batch,
        contexts,
        compute_forces=True,
        compute_stress=True,
        create_graph=True,
        return_aux=True,
    )
    loss = compute_potential_loss(output, differentiable_batch, loss_config)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert all(
        auxiliary["ot"].support_diagnostics.backend == "edge_list"
        and not auxiliary["ot"].dense_plan_materialized
        for auxiliary in output.auxiliary
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    optimizer_config = OptimizerConfig(learning_rate=1.0e-3, weight_decay=0.0)
    optimizer = build_optimizer(model, optimizer_config)
    step = train_step(
        model,
        optimizer,
        batch,
        contexts,
        loss_config,
        TrainStepConfig(gradient_clip_norm=100.0),
    )
    assert np.isfinite(step.total_loss) and step.number_of_parameters_with_grad > 0
    assert torch.equal(model.atomic_baseline, baseline)
    assert input_fingerprint == fingerprint_batch_sequence(
        (batch,), split_name="synthetic"
    )

    scheduler_config = SchedulerConfig(kind="none")
    scheduler = build_scheduler(optimizer, scheduler_config)
    selection = ModelSelectionState()
    progress = FitProgress(
        next_epoch=1,
        global_step=1,
        completed_epochs=1,
        last_completed_epoch=0,
    )
    train_step_config = TrainStepConfig(gradient_clip_norm=100.0)
    validation_step_config = ValidationStepConfig()
    selection_config = ModelSelectionConfig()
    fit_config = FitConfig(max_epochs=1)
    checkpoint = capture_training_checkpoint(
        model,
        optimizer,
        scheduler,
        selection,
        progress,
        (batch,),
        (batch,),
        model_config=model.config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        train_step_config=train_step_config,
        validation_step_config=validation_step_config,
        scheduler_config=scheduler_config,
        model_selection_config=selection_config,
        fit_config=fit_config,
        species_vocabulary=(6, 41),
    )
    path = tmp_path / "synthetic.pt"
    save_training_checkpoint(checkpoint, path)
    restored_checkpoint = load_training_checkpoint(path)
    assert torch.equal(restored_checkpoint.model_state_dict["atomic_baseline"], baseline)

    fresh = _model(template)
    fresh_optimizer = build_optimizer(fresh, optimizer_config)
    fresh_scheduler = build_scheduler(fresh_optimizer, scheduler_config)
    resumed = restore_training_checkpoint_(
        restored_checkpoint,
        fresh,
        fresh_optimizer,
        fresh_scheduler,
        (batch,),
        (batch,),
        contexts,
        {
            "model": fresh.config,
            "loss": loss_config,
            "optimizer": optimizer_config,
            "train_step": train_step_config,
            "validation_step": validation_step_config,
            "scheduler": scheduler_config,
            "model_selection": selection_config,
            "fit": FitConfig(max_epochs=2),
        },
        resumed_max_epochs=2,
        policy=ResumePolicy(),
    )
    assert resumed.exact_resume_ready and resumed.next_epoch == 1
    assert all(
        torch.equal(fresh.state_dict()[key], value)
        for key, value in restored_checkpoint.model_state_dict.items()
    )
