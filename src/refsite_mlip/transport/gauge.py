"""Gauge projection for balanced transport duals."""

from __future__ import annotations

import torch


def gauge_vector(
    num_rows: int, num_columns: int, reference: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        (
            torch.ones(num_rows, dtype=reference.dtype, device=reference.device),
            -torch.ones(
                num_columns, dtype=reference.dtype, device=reference.device
            ),
        )
    )


def project_gauge(
    vector: torch.Tensor, num_rows: int, num_columns: int
) -> torch.Tensor:
    if vector.shape[-1] != num_rows + num_columns:
        raise ValueError("gauge vector dimension mismatch")
    null = gauge_vector(num_rows, num_columns, vector)
    coefficient = torch.sum(vector * null, dim=-1, keepdim=True) / float(
        num_rows + num_columns
    )
    return vector - coefficient * null


def project_duals(
    f: torch.Tensor, g: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(f.shape[-1])
    columns = int(g.shape[-1])
    projected = project_gauge(torch.cat((f, g), dim=-1), rows, columns)
    return projected[..., :rows], projected[..., rows:]
