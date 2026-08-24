"""Static and runtime reciprocal-mode validation."""

from __future__ import annotations

import torch


def static_mode_amplitudes(
    reference_fields: torch.Tensor, channel_weights: torch.Tensor
) -> torch.Tensor:
    return torch.sum(
        channel_weights.unsqueeze(0) * reference_fields.abs().square(), dim=-1
    )


def runtime_atomic_mode_amplitudes(
    atomic_fields: torch.Tensor, channel_weights: torch.Tensor
) -> torch.Tensor:
    return torch.sum(channel_weights * atomic_fields.abs().square(), dim=-1)


def validate_static_mode_amplitudes(
    reference_fields: torch.Tensor,
    channel_weights: torch.Tensor,
    minimum_amplitude: float,
) -> torch.Tensor:
    amplitudes = static_mode_amplitudes(reference_fields, channel_weights)
    if not bool(torch.all(torch.isfinite(amplitudes))):
        raise ValueError("static typed mode amplitudes are non-finite")
    if bool(torch.any(amplitudes <= minimum_amplitude)):
        raise ValueError("static typed reciprocal mode is extinct")
    return amplitudes


def validate_runtime_amplitudes(
    atomic_fields: torch.Tensor,
    cross: torch.Tensor,
    channel_weights: torch.Tensor,
    minimum_atomic_amplitude: float,
    minimum_cross_amplitude: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    atomic_amplitude = runtime_atomic_mode_amplitudes(
        atomic_fields, channel_weights
    )
    if not bool(torch.all(torch.isfinite(atomic_amplitude))) or not bool(
        torch.all(torch.isfinite(cross.abs()))
    ):
        raise ValueError("runtime typed reciprocal amplitudes are non-finite")
    if bool(torch.any(atomic_amplitude <= minimum_atomic_amplitude)):
        raise ValueError("runtime atomic reciprocal amplitude collapsed")
    if bool(torch.any(cross.abs() <= minimum_cross_amplitude)):
        raise ValueError("runtime typed cross amplitude collapsed")
    return atomic_amplitude, cross.abs()
