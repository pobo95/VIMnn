from __future__ import annotations

import torch

from conftest import reciprocal_fields
from refsite_mlip.phase.initialization import primary_phase_initialization
from refsite_mlip.phase.stabilizer import torus_difference


def _initialize(data, positions=None, origin=None):
    _, _, cross = reciprocal_fields(data, positions=positions, origin=origin)
    return primary_phase_initialization(cross[:3], data["modes"][:3])


def test_nonzero_origin_and_joint_translation_are_invariant(typed_crystal):
    baseline = _initialize(typed_crystal)
    translation = torch.tensor([1.17, -0.63, 0.44], dtype=torch.float64)
    translated = _initialize(
        typed_crystal,
        positions=typed_crystal["positions"] + translation,
        origin=typed_crystal["origin"] + translation,
    )
    torch.testing.assert_close(
        torus_difference(translated, baseline),
        torch.zeros(3, dtype=torch.float64),
        atol=2.0e-13,
        rtol=0.0,
    )


def test_atom_only_arbitrary_cartesian_translation_is_covariant(typed_crystal):
    baseline = _initialize(typed_crystal)
    translation = torch.tensor([0.91, -1.24, 0.37], dtype=torch.float64)
    translated = _initialize(
        typed_crystal, positions=typed_crystal["positions"] + translation
    )
    fractional_translation = torch.linalg.solve(
        typed_crystal["cell"].T, translation
    )
    error = torus_difference(translated - baseline, fractional_translation)
    torch.testing.assert_close(
        error, torch.zeros_like(error), atol=3.0e-13, rtol=0.0
    )


def test_lattice_vector_translation_changes_only_integer_representative(typed_crystal):
    baseline = _initialize(typed_crystal)
    lattice_integer = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    translation = lattice_integer @ typed_crystal["cell"]
    translated = _initialize(
        typed_crystal, positions=typed_crystal["positions"] + translation
    )
    torch.testing.assert_close(
        torus_difference(translated, baseline),
        torch.zeros(3, dtype=torch.float64),
        atol=5.0e-13,
        rtol=0.0,
    )
