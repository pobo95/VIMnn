from __future__ import annotations

import torch

from refsite_mlip.features import ProbabilityMultipoleConfig
from refsite_mlip.interactions import (
    HigherBodyConfig,
    SymmetricCorrelationConfig,
)
from refsite_mlip.interactions.higher_body import (
    SYMMETRIC_POWER_CONTRACT_VERSION,
)
from refsite_mlip.models import PotentialConfig, ReferenceSitePotential

from test_grouped_template_batch import _case as _legacy_grouped_case


ANGULAR = "0e + 1o + 2e"
FEATURE_IRREPS = "2x0e+4x0e+4x1o+4x2e"
AVG_NUM_NEIGHBORS = 6.0


def v2_configuration(dtype, *, order=3, layers=2, channels=1):
    tolerance = 1.0e-6 if dtype == torch.float32 else 1.0e-7
    feature = ProbabilityMultipoleConfig(
        (6, 41),
        2,
        2,
        1.0,
        3.0,
        tolerance,
        site_type_vocabulary=(0, 1),
    )
    higher = HigherBodyConfig(
        FEATURE_IRREPS,
        2,
        2,
        site_type_embedding_dim=2,
        n_correlation_channels=channels,
        lmax=2,
        radial_feature_dim=3,
        radial_hidden_dims=(4,),
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        cutoff=3.0,
        edge_length_scale=1.0,
        correlation_mode=None,
        contract_version=SYMMETRIC_POWER_CONTRACT_VERSION,
        symmetric_correlation=SymmetricCorrelationConfig(order),
    )
    return PotentialConfig(
        (6, 41),
        layers,
        feature,
        higher,
        readout_hidden=8,
        energy_scale=1.0,
    )


def v2_grouped_case(
    typed_crystal,
    *,
    order=3,
    layers=2,
    channels=1,
    dtype=torch.float64,
    device="cpu",
    seed=117,
):
    data, _, registry, samples, batch, contexts = _legacy_grouped_case(
        typed_crystal, dtype=dtype, device=device
    )
    default = registry.resolve("zeta")
    torch.manual_seed(seed)
    model = ReferenceSitePotential(
        v2_configuration(
            dtype, order=order, layers=layers, channels=channels
        ),
        default.topology,
        default.phase_modes,
        default.phase_mode_weights,
        torch.eye(2, dtype=dtype),
        default.site_alignment_weights,
        default.phase_channel_weights,
        (-1.0, 2.0),
    ).to(device=device, dtype=dtype)
    return data, model, registry, samples, batch, contexts


def numbers(data, count):
    return torch.where(
        data["site_types"][:count] == 0,
        torch.tensor(6, dtype=torch.long, device=data["site_types"].device),
        torch.tensor(41, dtype=torch.long, device=data["site_types"].device),
    )


def direct_symmetric_message(layer, bank, density, central, *, order=None):
    contraction = layer.symmetric_contraction
    pieces = []
    start = 0
    for multiplicity, irrep in contraction.input_irreps:
        stop = start + multiplicity * irrep.dim
        pieces.append(
            density[:, start:stop].reshape(
                density.shape[0], multiplicity, irrep.dim
            )
        )
        start = stop
    packed = torch.cat(pieces, dim=-1)
    orders = (
        range(1, contraction.correlation_order + 1)
        if order is None
        else (order,)
    )
    blocks = []
    for output_index, (_, output_irrep) in enumerate(
        contraction.requested_output_irreps
    ):
        total = None
        for current in orders:
            basis = bank.basis_tensor(current, str(output_irrep))
            weight = contraction.weight_parameter(output_index, current)
            if current == 1:
                value = torch.einsum(
                    "sq,qpk,poa,ska->sko",
                    central,
                    weight,
                    basis,
                    packed,
                )
            elif current == 2:
                value = torch.einsum(
                    "sq,qpk,poab,ska,skb->sko",
                    central,
                    weight,
                    basis,
                    packed,
                    packed,
                )
            else:
                value = torch.einsum(
                    "sq,qpk,poabc,ska,skb,skc->sko",
                    central,
                    weight,
                    basis,
                    packed,
                    packed,
                    packed,
                )
            total = value if total is None else total + value
        blocks.append(total.reshape(total.shape[0], -1))
    return torch.cat(blocks, dim=-1)
