from __future__ import annotations

import pytest
import torch

from refsite_mlip.phase.stabilizer import (
    find_typed_stabilizer,
    permutation_for_translation,
    validate_alias_matches_stabilizer,
)


def _twofold_template(types=(0, 0, 1, 1)):
    sites = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.25, 0.1], [0.5, 0.25, 0.1]],
        dtype=torch.float64,
    )
    return sites, torch.tensor(types, dtype=torch.long)


def test_primary_integer_alias_group_equals_typed_stabilizer():
    sites, types = _twofold_template()
    stabilizer = find_typed_stabilizer(sites, types)
    modes = torch.tensor([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.long)
    assert stabilizer.translations.shape[0] == 2
    validate_alias_matches_stabilizer(modes, stabilizer)


def test_rank_three_is_not_enough_when_typed_alias_differs():
    sites, types = _twofold_template(types=(0, 1, 1, 0))
    stabilizer = find_typed_stabilizer(sites, types)
    modes = torch.tensor([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="alias group"):
        validate_alias_matches_stabilizer(modes, stabilizer)


def test_stabilizer_induces_site_permutation_for_probability_fields():
    sites, types = _twofold_template()
    stabilizer = find_typed_stabilizer(sites, types)
    tau = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    permutation = permutation_for_translation(tau, stabilizer)
    torch.testing.assert_close(permutation, torch.tensor([1, 0, 3, 2]))

    probability = torch.tensor(
        [[0.8, 0.1], [0.2, 0.7], [0.6, 0.3], [0.4, 0.9]],
        dtype=torch.float64,
    )
    vacancy = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=torch.float64)
    shifted_probability = probability.index_select(0, permutation)
    shifted_vacancy = vacancy.index_select(0, permutation)
    torch.testing.assert_close(shifted_probability, probability[permutation])
    torch.testing.assert_close(shifted_vacancy, vacancy[permutation])
