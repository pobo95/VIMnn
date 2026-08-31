from __future__ import annotations
import contextlib
import math
import torch
from torch import nn
from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.features import build_probability_multipoles,build_sparse_probability_multipoles
from refsite_mlip.geometry.reference import aligned_reference_sites
from refsite_mlip.graph import update_reference_edge_geometry
from refsite_mlip.interactions import CentralConditioner,EquivariantNodeEncoder,squared_edge_radial_basis
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.evaluation import solve_evaluation_phase
from refsite_mlip.phase.modes import (
    runtime_atomic_mode_amplitudes,
    static_mode_amplitudes,
    validate_runtime_amplitudes,
    validate_static_mode_amplitudes,
)
from refsite_mlip.phase.newton import solve_training_phase
from refsite_mlip.phase.objective import typed_reciprocal_fields
from refsite_mlip.phase.stabilizer import validate_alias_matches_stabilizer
from refsite_mlip.phase.types import EvaluationPhaseError
from refsite_mlip.transport import TRAIN_FIXED,EVAL_ADAPTIVE,TrainSinkhornConfig,EvalOTConfig,TransportSupportError,atom_site_displacements,build_compact_transport_edges,solve_atom_vacancy_ot,solve_sparse_sinkhorn_train_fixed
from .config import PotentialConfig
from .evaluation_policy import EvaluationPolicy
from .outputs import EvaluationDiagnostics, PotentialOutput
from .readout import SiteEnergyReadout
from .residual_block import ResidualInteractionBlock
from .template_context import TemplateExecutionContext

class ReferenceSitePotential(nn.Module):
    def __init__(self,config:PotentialConfig,topology,phase_modes,phase_mode_weights,species_alignment_weights,site_alignment_weights,phase_channel_weights,atomic_baseline=None):
        super().__init__(); config.validate(); self.config=config; self.topology=topology
        self.register_buffer('phase_modes',phase_modes); self.register_buffer('phase_mode_weights',phase_mode_weights); self.register_buffer('species_alignment_weights',species_alignment_weights); self.register_buffer('site_alignment_weights',site_alignment_weights); self.register_buffer('phase_channel_weights',phase_channel_weights)
        baseline=torch.zeros(len(config.species_vocabulary),dtype=topology.reference_cell.dtype) if atomic_baseline is None else torch.as_tensor(atomic_baseline,dtype=topology.reference_cell.dtype)
        if baseline.shape!=(len(config.species_vocabulary),): raise ValueError('atomic baseline shape mismatch')
        self.register_buffer('atomic_baseline',baseline)
        _,o3=import_e3nn_0_4_4(); higher=config.higher_body; self.central=CentralConditioner(higher.species_count,higher.site_type_count,higher.site_type_embedding_dim)
        self.irreps_hidden=o3.Irreps([(higher.n_correlation_channels,o3.Irrep(l,1 if l%2==0 else -1)) for l in range(higher.lmax+1)])
        self.probability_encoder=EquivariantNodeEncoder(config.feature_irreps,self.irreps_hidden); self.central_encoder=o3.Linear(self.central.irreps,self.irreps_hidden,biases=False)
        beta=1/math.sqrt(config.num_layers); self.layers=nn.ModuleList([ResidualInteractionBlock(self.irreps_hidden,self.central.irreps,higher,beta) for _ in range(config.num_layers)])
        scalar_dim=self.irreps_hidden[0].mul; self.scalar_slice=self.irreps_hidden.slices()[0]
        self.readout=SiteEnergyReadout(scalar_dim,self.central.num_channels,config.readout_hidden,config.energy_scale)
    def _species_indices(self,atomic_numbers):
        vocab=torch.tensor(self.config.species_vocabulary,dtype=torch.long,device=atomic_numbers.device); match=atomic_numbers[:,None]==vocab[None,:]
        if bool(torch.any(match.sum(-1)!=1)): raise ValueError('unknown atomic species')
        return torch.argmax(match.to(torch.long),-1)
    def _resolve_template_context(self,context,positions,atomic_numbers):
        if not isinstance(context,TemplateExecutionContext): raise TypeError('template_context must be a TemplateExecutionContext')
        context.validate_fingerprint(); topology=context.topology; higher=self.config.higher_body
        if context.convention_version!='reference_template_v1': raise ValueError('unsupported template convention version')
        if tuple(topology.pbc)!=(True,True,True): raise ValueError('runtime template must be fully periodic')
        if not set(context.supported_species).issubset(set(self.config.species_vocabulary)): raise ValueError('template supported species are incompatible with model vocabulary')
        if topology.site_types.numel() and bool(torch.any((topology.site_types<0)|(topology.site_types>=higher.site_type_count))): raise ValueError('runtime template contains an unknown global site type')
        modes=context.phase_modes; weights=context.phase_mode_weights
        if modes.ndim!=2 or modes.shape[1]!=3 or modes.dtype!=torch.long or modes.shape[0]<3: raise ValueError('runtime template requires at least three long phase modes')
        if weights.shape!=(modes.shape[0],): raise ValueError('runtime phase mode/weight shape mismatch')
        if abs(float(torch.linalg.det(modes[:3].to(torch.float64))))<=1.0e-12: raise ValueError('runtime primary phase modes must be nonsingular')
        M=topology.num_sites
        if positions.shape[0]>M: raise ValueError(f'atom count N={positions.shape[0]} exceeds reference-site count M={M}')
        actual_species=set(int(z) for z in atomic_numbers.detach().cpu().tolist())
        if not actual_species.issubset(set(context.supported_species)): raise ValueError('atomic species are unsupported by runtime template')
        if context.site_alignment_weights.ndim!=2 or context.site_alignment_weights.shape[0]!=M: raise ValueError('runtime site alignment must have shape [M,C]')
        C=context.site_alignment_weights.shape[1]
        if context.phase_channel_weights.shape!=(C,): raise ValueError('runtime phase channel weights must have shape [C]')
        if self.species_alignment_weights.shape!=(len(self.config.species_vocabulary),C): raise ValueError('runtime phase channel count is incompatible with model species alignment')
        if not math.isclose(topology.cutoff,higher.cutoff,rel_tol=0.0,abs_tol=1.0e-12): raise ValueError('runtime topology cutoff is incompatible with model cutoff')
        if not math.isclose(context.avg_num_neighbors,higher.avg_num_neighbors,rel_tol=0.0,abs_tol=1.0e-12): raise ValueError('runtime avg_num_neighbors convention is incompatible with model')
        vocabulary=self.config.feature.site_type_vocabulary
        if vocabulary is not None and topology.site_types.numel() and not set(topology.site_types.tolist()).issubset(set(vocabulary)): raise ValueError('runtime template site types are outside the model feature vocabulary')
        return context.materialize(device=positions.device,dtype=positions.dtype)
    def _evaluation_preflight(
        self,
        policy,
        runtime,
        atomic_fields,
        reference_fields,
        cross,
    ):
        policy.validate_fingerprint()
        if (
            policy.template_id != runtime.template_id
            or policy.template_fingerprint != runtime.fingerprint
        ):
            raise EvaluationPhaseError(
                "POLICY_CONTEXT_MISMATCH",
                "evaluation policy and template context binding differ",
                template_id=runtime.template_id,
            )
        stabilizer = runtime.stabilizer
        translations = stabilizer.translations
        permutations = stabilizer.permutations
        expected_permutation = torch.arange(
            runtime.topology.num_sites,
            dtype=torch.long,
            device=permutations.device,
        )
        stabilizer_valid = (
            translations.ndim == 2
            and translations.shape[1:] == (3,)
            and translations.shape[0] > 0
            and permutations.shape
            == (translations.shape[0], runtime.topology.num_sites)
            and permutations.dtype == torch.long
            and bool(torch.all(torch.isfinite(translations)))
            and all(
                torch.equal(torch.sort(row).values, expected_permutation)
                for row in permutations
            )
        )
        if not stabilizer_valid:
            raise EvaluationPhaseError(
                "ALIAS_STABILIZER_MISMATCH",
                "typed stabilizer metadata is invalid",
                template_id=runtime.template_id,
            )
        try:
            validate_alias_matches_stabilizer(
                runtime.phase_modes[:3], stabilizer, policy.equivalence_tolerance
            )
            validate_alias_matches_stabilizer(
                runtime.phase_modes, stabilizer, policy.equivalence_tolerance
            )
        except ValueError as error:
            raise EvaluationPhaseError(
                "ALIAS_STABILIZER_MISMATCH",
                str(error),
                template_id=runtime.template_id,
            ) from error

        reference_amplitudes = static_mode_amplitudes(
            reference_fields, runtime.phase_channel_weights
        )
        try:
            validate_static_mode_amplitudes(
                reference_fields,
                runtime.phase_channel_weights,
                policy.minimum_reference_amplitude_absolute,
            )
        except ValueError as error:
            raise EvaluationPhaseError(
                "REFERENCE_MODE_EXTINCTION",
                str(error),
                template_id=runtime.template_id,
                observed=float(reference_amplitudes.detach().min().cpu()),
                threshold=policy.minimum_reference_amplitude_absolute,
            ) from error

        atomic_amplitudes = runtime_atomic_mode_amplitudes(
            atomic_fields, runtime.phase_channel_weights
        )
        cross_amplitudes = cross.abs()
        try:
            validate_runtime_amplitudes(
                atomic_fields,
                cross,
                runtime.phase_channel_weights,
                policy.minimum_atomic_amplitude_absolute,
                policy.minimum_cross_amplitude_absolute,
            )
        except ValueError as error:
            atomic_failed = (
                not bool(torch.all(torch.isfinite(atomic_amplitudes)))
                or bool(
                    torch.any(
                        atomic_amplitudes
                        <= policy.minimum_atomic_amplitude_absolute
                    )
                )
            )
            reason = (
                "ATOMIC_MODE_EXTINCTION"
                if atomic_failed
                else "CROSS_AMPLITUDE_TOO_SMALL"
            )
            observed = (
                atomic_amplitudes.detach().min().cpu()
                if atomic_failed
                else cross_amplitudes.detach().min().cpu()
            )
            threshold = (
                policy.minimum_atomic_amplitude_absolute
                if atomic_failed
                else policy.minimum_cross_amplitude_absolute
            )
            raise EvaluationPhaseError(
                reason,
                str(error),
                template_id=runtime.template_id,
                observed=float(observed),
                threshold=threshold,
            ) from error
        return (
            policy.materialize_candidate_offsets(
                device=cross.device, dtype=cross.real.dtype
            ),
            atomic_amplitudes,
            reference_amplitudes,
            cross_amplitudes,
        )

    @staticmethod
    def _compact_evaluation_diagnostics(
        runtime,
        policy,
        evaluation,
        atomic_amplitudes,
        reference_amplitudes,
        cross_amplitudes,
        ot,
        effective_transport_tolerance,
        support_config,
        *,
        derivative_requested=False,
        forces_requested=False,
        stress_requested=False,
    ):
        def scalar(value):
            return float(value.detach().cpu())

        curvature = torch.linalg.eigvalsh(-evaluation.refined.hessian)
        minimum_curvature = curvature[..., 0]
        maximum_curvature = curvature[..., -1]
        support = ot.support_diagnostics
        compact = support_config.kind == "compact_c2"
        expected_q_mass = runtime.topology.num_sites - ot.P.shape[1]
        return EvaluationDiagnostics(
            template_id=runtime.template_id,
            template_fingerprint=runtime.fingerprint,
            context_fingerprint=runtime.fingerprint,
            policy_template_fingerprint=policy.template_fingerprint,
            policy_content_fingerprint=policy.content_fingerprint,
            input_candidate_count=evaluation.input_candidate_count,
            non_equivalent_group_count=evaluation.non_equivalent_group_count,
            selected_original_candidate_index=int(
                evaluation.selected_index.detach().cpu()
            ),
            selected_grouped_index=int(
                evaluation.selected_grouped_index.detach().cpu()
            ),
            best_raw_score=scalar(evaluation.best_raw_score),
            second_best_non_equivalent_raw_score=scalar(
                evaluation.second_best_raw_score
            ),
            absolute_objective_gap=scalar(evaluation.non_equivalent_gap),
            selected_pre_refinement_phase=evaluation.selected_candidate.detach()
            .cpu()
            .clone(),
            refined_phase=evaluation.refined.phase.detach().cpu().clone(),
            minimum_atomic_amplitude=scalar(atomic_amplitudes.min()),
            minimum_cross_amplitude=scalar(cross_amplitudes.min()),
            minimum_reference_amplitude=scalar(reference_amplitudes.min()),
            final_gradient_norm=scalar(
                torch.linalg.vector_norm(evaluation.refined.gradient)
            ),
            hessian_minimum_curvature=scalar(minimum_curvature),
            hessian_maximum_curvature=scalar(maximum_curvature),
            hessian_condition_number=scalar(
                maximum_curvature / minimum_curvature
            ),
            stabilizer_size=int(runtime.stabilizer.translations.shape[0]),
            alias_stabilizer_validated=True,
            phase_failure_reason=None,
            transport_path=ot.path_name,
            transport_solver_name=ot.solver_name,
            transport_row_residual=scalar(ot.row_residual),
            transport_column_residual=scalar(ot.column_residual),
            transport_sinkhorn_iterations=ot.sinkhorn_iterations,
            transport_sinkhorn_warmup_iterations=ot.warmup_sinkhorn_iterations,
            transport_fallback_sinkhorn_iterations=ot.fallback_sinkhorn_iterations,
            transport_newton_iterations=ot.newton_iterations,
            transport_cg_iterations=ot.cg_iterations,
            transport_fallback_used=ot.fallback_used,
            transport_fallback_reason=ot.failure_reason,
            transport_kind=support_config.kind,
            transport_r_on=support_config.r_on if compact else None,
            transport_r_off=support_config.cutoff if compact else None,
            transport_r_candidate=(
                support_config.r_candidate if compact else None
            ),
            transport_core_edge_count=(
                support.core_edge_count if support is not None else None
            ),
            transport_active_edge_count=(
                support.active_edge_count if support is not None else None
            ),
            transport_candidate_edge_count=(
                support.candidate_edge_count if support is not None else None
            ),
            transport_maximum_matching_size=(
                support.maximum_atom_matching_size
                if support is not None
                else None
            ),
            transport_total_support_feasible=(
                support.total_support_feasible if support is not None else None
            ),
            transport_switch_on_boundary_gap=(
                support.switch_on_boundary_gap if support is not None else None
            ),
            transport_cutoff_boundary_gap=(
                support.cutoff_boundary_gap if support is not None else None
            ),
            transport_candidate_boundary_gap=(
                support.candidate_boundary_gap if support is not None else None
            ),
            transport_line_search_reductions=ot.line_search_reductions,
            transport_accepted_damping=ot.accepted_damping,
            transport_q_mass_error=scalar(
                torch.abs(ot.q.sum() - ot.q.new_tensor(expected_q_mass))
            ),
            effective_transport_tolerance=effective_transport_tolerance,
            differentiability_scope=(
                "selected_branch_first_order"
                if derivative_requested
                else "energy_only"
            ),
            hard_branch_frozen=True,
            derivative_order=1 if derivative_requested else 0,
            forces_requested=forces_requested,
            stress_requested=stress_requested,
        )

    def energy(
        self,
        positions,
        atomic_numbers,
        cell,
        origin,
        *,
        solver_path=TRAIN_FIXED,
        return_aux=False,
        template_context=None,
        evaluation_policy=None,
        _evaluation_derivative_request=False,
        _forces_requested=False,
        _stress_requested=False,
    ):
        if solver_path == TRAIN_FIXED:
            if evaluation_policy is not None:
                raise ValueError(
                    "TRAIN_FIXED requires evaluation_policy=None"
                )
        elif solver_path == EVAL_ADAPTIVE:
            if evaluation_policy is None:
                raise ValueError(
                    "EVAL_ADAPTIVE now requires an explicit template-bound "
                    "evaluation_policy; fixed-phase evaluation is no longer implicit"
                )
            if template_context is None:
                raise ValueError(
                    "EVAL_ADAPTIVE requires a TemplateExecutionContext with a typed stabilizer"
                )
            if not isinstance(evaluation_policy, EvaluationPolicy):
                raise TypeError("evaluation_policy must be an EvaluationPolicy")
        else:
            raise ValueError("unsupported solver path")
        if (
            self.config.transport_support.backend == "edge_list"
            and solver_path == EVAL_ADAPTIVE
        ):
            raise TransportSupportError(
                "EDGE_LIST_EVAL_ADAPTIVE_UNSUPPORTED",
                "edge_list compact transport currently supports TRAIN_FIXED only; use backend=dense for EVAL_ADAPTIVE",
                template_id=getattr(template_context, "template_id", None),
            )
        runtime = None
        if template_context is None:
            topology=self.topology.to(device=positions.device,dtype=positions.dtype); phase_modes=self.phase_modes; phase_mode_weights=self.phase_mode_weights.to(positions); site_alignment_weights=self.site_alignment_weights.to(positions); phase_channel_weights=self.phase_channel_weights.to(positions)
        else:
            runtime=self._resolve_template_context(template_context,positions,atomic_numbers); topology=runtime.topology; phase_modes=runtime.phase_modes; phase_mode_weights=runtime.phase_mode_weights; site_alignment_weights=runtime.site_alignment_weights; phase_channel_weights=runtime.phase_channel_weights
        species=self._species_indices(atomic_numbers); atom_weights=self.species_alignment_weights.to(positions)[species]
        atomic_fields,reference_fields,cross=typed_reciprocal_fields(positions,origin,cell,topology.reference_fractional,atom_weights,site_alignment_weights,phase_modes,phase_channel_weights)
        initial=primary_phase_initialization(cross[:3],phase_modes[:3])
        evaluation = None
        amplitudes = None
        if solver_path == TRAIN_FIXED:
            phase=solve_training_phase(cross,phase_modes,phase_mode_weights,initial,self.config.phase_steps,self.config.phase_damping).phase
        else:
            candidate_offsets, atomic_amplitudes, reference_amplitudes, cross_amplitudes = self._evaluation_preflight(evaluation_policy,runtime,atomic_fields,reference_fields,cross)
            amplitudes = (atomic_amplitudes, reference_amplitudes, cross_amplitudes)
            try:
                evaluation=solve_evaluation_phase(
                    cross,
                    phase_modes,
                    phase_mode_weights,
                    initial,
                    candidate_offsets,
                    runtime.stabilizer,
                    evaluation_policy.phase_step_schedule,
                    evaluation_policy.phase_damping_schedule,
                    minimum_gap=evaluation_policy.minimum_objective_gap_absolute,
                    minimum_curvature=evaluation_policy.minimum_curvature,
                    maximum_condition=evaluation_policy.maximum_condition,
                    maximum_gradient_norm=evaluation_policy.maximum_gradient_norm,
                    minimum_cross_amplitude=evaluation_policy.minimum_cross_amplitude_absolute,
                    equivalence_tolerance=evaluation_policy.equivalence_tolerance,
                )
            except EvaluationPhaseError as error:
                if error.template_id is not None:
                    raise
                raise EvaluationPhaseError(
                    error.reason_code,
                    str(error),
                    template_id=runtime.template_id,
                    observed=error.observed,
                    threshold=error.threshold,
                ) from error
            phase=evaluation.refined.phase
        if (
            solver_path == EVAL_ADAPTIVE
            and _evaluation_derivative_request
            and not phase.requires_grad
        ):
            raise EvaluationPhaseError(
                "GRAPH_DISCONNECTED",
                "selected refined phase is detached from the input geometry",
                template_id=runtime.template_id,
            )
        references=aligned_reference_sites(topology.reference_fractional,phase,origin,cell); displacements=atom_site_displacements(positions,references,cell,topology.pbc)
        edge_backend=(
            self.config.transport_support.kind == "compact_c2"
            and self.config.transport_support.backend == "edge_list"
        )
        if edge_backend:
            edges=build_compact_transport_edges(
                displacements,
                epsilon_ot=self.config.epsilon_ot,
                ell_ot=self.config.ell_ot,
                config=self.config.transport_support,
                template_id=getattr(runtime, "template_id", None),
            )
            ot=solve_sparse_sinkhorn_train_fixed(
                edges,TrainSinkhornConfig(self.config.train_sinkhorn_iterations)
            )
        else:
            cost=displacements.square().sum(-1)/(2*self.config.ell_ot**2)
        if solver_path==TRAIN_FIXED and not edge_backend:
            distances = (
                torch.linalg.vector_norm(displacements, dim=-1)
                if self.config.transport_support.kind == "compact_c2"
                else None
            )
            ot=solve_atom_vacancy_ot(
                cost,self.config.epsilon_ot,TRAIN_FIXED,'sinkhorn',
                TrainSinkhornConfig(self.config.train_sinkhorn_iterations),
                support_config=self.config.transport_support,
                atom_distances=distances,
                template_id=getattr(runtime, "template_id", None),
            )
        elif solver_path==EVAL_ADAPTIVE:
            evaluation_ot_tolerance = (
                1.0e-6 if cost.dtype == torch.float32 else 1.0e-12
            )
            distances = (
                torch.linalg.vector_norm(displacements, dim=-1)
                if self.config.transport_support.kind == "compact_c2"
                else None
            )
            ot=solve_atom_vacancy_ot(
                cost,
                self.config.epsilon_ot,
                EVAL_ADAPTIVE,
                "hybrid",
                EvalOTConfig(
                    sinkhorn_iterations=self.config.eval_sinkhorn_warmup_iterations,
                    convergence_tolerance=evaluation_ot_tolerance,
                ),
                support_config=self.config.transport_support,
                atom_distances=distances,
                template_id=runtime.template_id,
            )
            if _evaluation_derivative_request and ot.fallback_used:
                raise EvaluationPhaseError(
                    "DERIVATIVE_FALLBACK_UNSUPPORTED",
                    "the selected adaptive transport fallback is not certified for derivatives",
                    template_id=runtime.template_id,
                )
            vacancy_present = topology.num_sites > positions.shape[0]
            if _evaluation_derivative_request and (
                not ot.P.requires_grad
                or (vacancy_present and not ot.q.requires_grad)
            ):
                raise EvaluationPhaseError(
                    "GRAPH_DISCONNECTED",
                    "adaptive transport P or q is detached from the selected geometry branch",
                    template_id=runtime.template_id,
                )
        if edge_backend:
            features=build_sparse_probability_multipoles(
                ot.edge_plan,ot.q,ot.edges,atomic_numbers,self.config.feature,topology.site_types
            )
        else:
            features=build_probability_multipoles(ot.P,ot.q,atomic_numbers,displacements,self.config.feature,topology.site_types)
        c_raw=features.raw_probability_state; c_bar=self.central(c_raw,topology.site_types); h=self.probability_encoder(features.equivariant_features)+self.central_encoder(c_bar)
        geometry=update_reference_edge_geometry(topology,cell,edge_length_scale=self.config.higher_body.edge_length_scale); radial=squared_edge_radial_basis(geometry.radial_coordinate,self.config.higher_body.radial_feature_dim)
        correlations=[]
        for layer in self.layers:
            h,corr=layer(h,c_bar,topology.edge_index,geometry.edge_vectors,radial,geometry.cutoff_values)
            if return_aux: correlations.append(corr)
        site_energy=self.readout(h[:,self.scalar_slice],c_bar); residual=site_energy.sum(); baseline=self.atomic_baseline.to(positions)[species].sum(); total=baseline+residual
        if solver_path == EVAL_ADAPTIVE and not bool(torch.isfinite(total).detach()):
            raise EvaluationPhaseError(
                "NONFINITE_OUTPUT",
                "evaluation energy is nonfinite",
                template_id=runtime.template_id,
            )
        if return_aux:
            aux={'phase':phase,'ot':ot,'q':ot.q,'multipoles':features,'correlations':correlations}
            if ot.support_diagnostics is not None:
                aux['transport_support']=ot.support_diagnostics
            if evaluation is not None:
                aux['evaluation_diagnostics']=self._compact_evaluation_diagnostics(
                    runtime,
                    evaluation_policy,
                    evaluation,
                    *amplitudes,
                    ot,
                    evaluation_ot_tolerance,
                    self.config.transport_support,
                    derivative_requested=_evaluation_derivative_request,
                    forces_requested=_forces_requested,
                    stress_requested=_stress_requested,
                )
        else:
            aux=None
        return PotentialOutput(total,site_energy,baseline,residual,h,c_raw,auxiliary=aux)
    def forward(self,positions,atomic_numbers,cell,origin,*,solver_path=TRAIN_FIXED,compute_forces=False,compute_stress=False,create_graph=False,return_aux=False,template_context=None,evaluation_policy=None):
        derivative_requested = compute_forces or compute_stress
        if solver_path == EVAL_ADAPTIVE and create_graph:
            raise EvaluationPhaseError(
                "CREATE_GRAPH_UNSUPPORTED",
                "EVAL_ADAPTIVE supports selected-branch first derivatives only; create_graph=True is unsupported",
                template_id=getattr(template_context, "template_id", None),
            )
        if (
            solver_path == EVAL_ADAPTIVE
            and derivative_requested
            and torch.is_inference_mode_enabled()
        ):
            raise EvaluationPhaseError(
                "INFERENCE_MODE_DERIVATIVE_UNSUPPORTED",
                "EVAL_ADAPTIVE force/stress requires autograd and cannot run under torch.inference_mode()",
                template_id=getattr(template_context, "template_id", None),
            )
        if not derivative_requested:
            return self.energy(positions,atomic_numbers,cell,origin,solver_path=solver_path,return_aux=return_aux,template_context=template_context,evaluation_policy=evaluation_policy)

        grad_context = (
            torch.enable_grad()
            if solver_path == EVAL_ADAPTIVE
            else contextlib.nullcontext()
        )
        with grad_context:
            derivative_positions = positions
            if (
                solver_path == EVAL_ADAPTIVE
                and compute_forces
                and not derivative_positions.requires_grad
            ):
                derivative_positions = positions.detach().clone().requires_grad_(True)
            strain=torch.zeros((3,3),dtype=positions.dtype,device=positions.device,requires_grad=compute_stress)
            F=torch.eye(3,dtype=positions.dtype,device=positions.device)+strain
            p=derivative_positions@F; H=cell@F; o=origin@F
            out=self.energy(
                p,atomic_numbers,H,o,
                solver_path=solver_path,
                return_aux=return_aux,
                template_context=template_context,
                evaluation_policy=evaluation_policy,
                _evaluation_derivative_request=(solver_path == EVAL_ADAPTIVE),
                _forces_requested=compute_forces,
                _stress_requested=compute_stress,
            )
            inputs=[]
            if compute_forces: inputs.append(derivative_positions)
            if compute_stress: inputs.append(strain)
            try:
                gradients=torch.autograd.grad(
                    out.energy,
                    inputs,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                )
            except RuntimeError as error:
                if solver_path != EVAL_ADAPTIVE:
                    raise
                raise EvaluationPhaseError(
                    "GRAPH_DISCONNECTED",
                    "selected evaluation branch is not differentiably connected to requested geometry",
                    template_id=getattr(template_context, "template_id", None),
                ) from error
            index=0; forces=None; stress=None; voigt=None
            if compute_forces: forces=-gradients[index]; index+=1
            if compute_stress:
                stress=gradients[index]/torch.linalg.det(cell).abs(); stress=.5*(stress+stress.T); voigt=stress[(0,1,2,1,0,0),(0,1,2,2,2,1)]
            if solver_path == EVAL_ADAPTIVE:
                if forces is not None and not bool(torch.all(torch.isfinite(forces)).detach()):
                    raise EvaluationPhaseError(
                        "NONFINITE_OUTPUT",
                        "evaluation forces are nonfinite",
                        template_id=getattr(template_context, "template_id", None),
                    )
                if stress is not None and not bool(torch.all(torch.isfinite(stress)).detach()):
                    raise EvaluationPhaseError(
                        "NONFINITE_OUTPUT",
                        "evaluation stress is nonfinite",
                        template_id=getattr(template_context, "template_id", None),
                    )
            return PotentialOutput(out.energy,out.site_energy,out.baseline_energy,out.residual_energy,out.site_features,out.raw_c,forces,stress,voigt,out.auxiliary)
