from __future__ import annotations

import math

import torch
from torch import nn

from refsite_mlip.compatibility import import_e3nn_0_4_4
from refsite_mlip.interactions import (
    CentralOuterProduct,
    DensityCorrelations,
    EdgeNeighborDensity,
)
from refsite_mlip.interactions.higher_body import (
    LEGACY_HIGHER_BODY_CONTRACT_VERSION,
    SYMMETRIC_POWER_CONTRACT_VERSION,
)
from refsite_mlip.interactions.symmetric_cg import SymmetricCGBasisBank
from refsite_mlip.interactions.symmetric_contraction import (
    FactorizedSymmetricContraction,
)


import_e3nn_0_4_4()
from e3nn.nn import Gate


class ResidualInteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_irreps,
        central_irreps,
        higher_config,
        residual_scale,
        basis_bank=None,
    ):
        super().__init__()
        _, o3 = import_e3nn_0_4_4()
        self.irreps_h = o3.Irreps(hidden_irreps)
        self.residual_scale = float(residual_scale)
        self.contract_version = higher_config.contract_version
        parsed_central = None
        representative = None
        if self.contract_version == SYMMETRIC_POWER_CONTRACT_VERSION:
            higher_config.validate()
            # Validate every external architecture fact before the first
            # parameter-producing module is constructed.
            if not isinstance(basis_bank, SymmetricCGBasisBank):
                raise ValueError(
                    "v2 residual block requires an external "
                    "SymmetricCGBasisBank"
                )
            parsed_central = o3.Irreps(central_irreps)
            if any(irrep.l != 0 or irrep.p != 1 for _, irrep in parsed_central):
                raise ValueError(
                    "v2 central conditioning must contain only even scalar "
                    "channels"
                )
            expected_central_dimension = (
                2
                + higher_config.species_count
                + 2 * higher_config.site_type_embedding_dim
            )
            if parsed_central.dim != expected_central_dimension:
                raise ValueError(
                    "v2 central conditioning dimension must contain constant, "
                    "species, vacancy, site-type, and vacancy-site-type channels"
                )
            symmetric = higher_config.symmetric_correlation
            if (
                basis_bank.basis_kind != symmetric.basis_kind
                or basis_bank.basis_version != symmetric.basis_version
                or basis_bank.normalization != symmetric.normalization
                or basis_bank.correlation_order
                != symmetric.correlation_order
            ):
                raise ValueError(
                    "external symmetric-CG basis does not match v2 "
                    "correlation config"
                )
            coupling_irreps = o3.Irreps(
                [(1, irrep) for _, irrep in self.irreps_h]
            )
            if (
                basis_bank.input_irreps != str(coupling_irreps)
                or basis_bank.requested_output_irreps
                != str(coupling_irreps)
            ):
                raise ValueError(
                    "external symmetric-CG basis irreps do not match hidden "
                    "irreps"
                )
            representative = basis_bank.basis_tensor(
                1, str(coupling_irreps[0].ir)
            )
        self.edge = EdgeNeighborDensity(
            self.irreps_h + central_irreps,
            self.irreps_h,
            higher_config.lmax,
            higher_config.radial_feature_dim,
            higher_config.radial_hidden_dims,
            higher_config.avg_num_neighbors,
        )

        if self.contract_version == LEGACY_HIGHER_BODY_CONTRACT_VERSION:
            # Keep the v1 module construction and registration order exact.
            self.corr = DensityCorrelations(
                self.irreps_h, higher_config.correlation_mode
            )
            self.outer = nn.ModuleList(
                [
                    CentralOuterProduct(central_irreps, self.irreps_h)
                    for _ in range(3)
                ]
            )
            self.contract = nn.ModuleList(
                [
                    o3.Linear(item.irreps_out, self.irreps_h, biases=False)
                    for item in self.outer
                ]
            )
        elif self.contract_version == SYMMETRIC_POWER_CONTRACT_VERSION:
            self.symmetric_contraction = (
                FactorizedSymmetricContraction.from_basis_bank(
                    self.irreps_h,
                    central_dimension=parsed_central.dim,
                    basis_bank=basis_bank,
                    dtype=representative.dtype,
                    device=representative.device,
                )
            )
            # Plain immutable identity only: registering the bank here would make
            # every residual layer another persistent owner of U.
            self.symmetric_basis_fingerprint = basis_bank.basis_fingerprint
        else:
            raise ValueError("unsupported higher-body contract version")

        scalars = []
        gated = []
        for multiplicity, irrep in self.irreps_h:
            if irrep.l == 0:
                scalars.append((multiplicity, irrep))
            else:
                gated.append((multiplicity, irrep))
        self.irreps_scalars = o3.Irreps(scalars)
        self.irreps_gated = o3.Irreps(gated)
        self.irreps_gates = (
            o3.Irreps(
                [(sum(multiplicity for multiplicity, _ in gated), o3.Irrep("0e"))]
            )
            if gated
            else o3.Irreps("")
        )
        acts = [
            torch.nn.functional.silu if irrep.p == 1 else torch.tanh
            for _, irrep in self.irreps_scalars
        ]
        gate_acts = [torch.sigmoid for _ in self.irreps_gates]
        self.gate = Gate(
            self.irreps_scalars,
            acts,
            self.irreps_gates,
            gate_acts,
            self.irreps_gated,
        )
        if self.gate.irreps_out != self.irreps_h:
            raise ValueError("Gate output must equal hidden irreps")
        gate_in = self.gate.irreps_in
        self.self_projection = o3.Linear(
            self.irreps_h, gate_in, biases=False
        )
        self.message_projection = o3.Linear(
            self.irreps_h, gate_in, biases=False
        )
        self.raw_skip = o3.Linear(
            central_irreps, self.irreps_h, biases=False
        )

    def forward(
        self,
        h,
        c_bar,
        edge_index,
        edge_vectors,
        edge_radial,
        edge_cutoff,
        *,
        symmetric_cg_basis=None,
    ):
        source = torch.cat((h, c_bar), dim=-1)
        _, _, _, density = self.edge(
            source,
            edge_index,
            edge_vectors,
            edge_radial,
            edge_cutoff,
            h.shape[0],
        )
        if self.contract_version == LEGACY_HIGHER_BODY_CONTRACT_VERSION:
            C1, C2, C3 = self.corr(density)
            products = [
                self.outer[index](c_bar, correlation)
                for index, correlation in enumerate((C1, C2, C3))
            ]
            message = sum(
                self.contract[index](products[index]) for index in range(3)
            ) / math.sqrt(3.0)
            correlation_details = {
                "A": density,
                "C1": C1,
                "C2": C2,
                "C3": C3,
                "Z1": products[0],
                "Z2": products[1],
                "Z3": products[2],
            }
        else:
            symmetric = self.symmetric_contraction(
                density,
                c_bar,
                basis_bank=symmetric_cg_basis,
            )
            message = symmetric.output
            correlation_details = {
                "A": density,
                "symmetric_output": message,
                "correlation_order": symmetric.correlation_order,
                "basis_fingerprint": self.symmetric_basis_fingerprint,
                "basis_kind": symmetric.diagnostics.basis_kind,
                "dense_A_outer_materialized": (
                    symmetric.diagnostics.dense_A_outer_materialized
                ),
            }
        delta = self.gate(
            self.self_projection(h) + self.message_projection(message)
        )
        updated = (
            h
            + self.residual_scale * delta
            + self.raw_skip(c_bar)
        )
        return updated, correlation_details
