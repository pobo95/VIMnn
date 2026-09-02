from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from refsite_mlip.cli.errors import CLIError, format_cli_error
from refsite_mlip.cli.inspect_bundle import (
    render_human,
    render_json,
    summarize_bundle,
)
from refsite_mlip.cli.main import build_parser, main


class _Phase:
    convention_version = "phase-test-v1"

    def __init__(self, *, reverse: bool) -> None:
        pairs = [
            ("modes", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ("mode_weights", [1.0, 1.0, 1.0]),
            ("site_type_alignment_weights", [[1.0, 0.0], [0.0, 1.0]]),
            ("channel_weights", [1.0, 1.25]),
            ("approval_status", "provisional"),
            ("convention_version", self.convention_version),
            ("floating_dtype", "float64"),
        ]
        self._payload = dict(reversed(pairs) if reverse else pairs)

    def to_dict(self):
        return dict(self._payload)


def _mapping(pairs, *, reverse: bool):
    return dict(reversed(pairs) if reverse else pairs)


def _binding(template_id: str, *, policy: bool, reverse: bool):
    return SimpleNamespace(
        template_id=template_id,
        approval_status="provisional",
        evaluation_policy=(
            SimpleNamespace(content_fingerprint="d" * 64) if policy else None
        ),
        full_template_fingerprint=("b" if template_id == "alpha" else "c") * 64,
        phase_specification=_Phase(reverse=reverse),
        provenance=_mapping(
            [("source", "synthetic"), ("ordinal", 1 if template_id == "alpha" else 2)],
            reverse=reverse,
        ),
        structural_artifact=SimpleNamespace(
            structural_fingerprint=("8" if template_id == "alpha" else "9") * 64
        ),
    )


def _fake_bundle(*, reverse: bool):
    bindings = [
        _binding("alpha", policy=True, reverse=reverse),
        _binding("zeta", policy=False, reverse=reverse),
    ]
    if reverse:
        bindings.reverse()
    return SimpleNamespace(
        architecture_fingerprint="a" * 64,
        bundle_fingerprint="f" * 64,
        bundle_scope="portable_reference_site_potential",
        conventions=_mapping(
            [
                ("convention_version", "bundle-conventions-v1"),
                ("ordered_species_vocabulary", [6, 41]),
                ("ordered_site_type_vocabulary", [0, 1]),
                ("phase_channel_count", 2),
                ("species_alignment_weights", torch.eye(2, dtype=torch.float64)),
                ("length_unit", "angstrom"),
                ("energy_unit", "eV"),
                ("force_unit", "eV/angstrom"),
                ("stress_unit", "eV/angstrom^3"),
                ("stress_sign", "tensile_positive"),
                ("stress_voigt_order", ["xx", "yy", "zz", "yz", "xz", "xy"]),
                ("cell_convention", "row_vector"),
                ("pbc_convention", "full_3d"),
                ("atomic_baseline_convention", "frozen_model_buffer_v1"),
                ("unit_convention_version", "angstrom_ev_tensile_voigt_v1"),
            ],
            reverse=reverse,
        ),
        default_template_id="zeta",
        model_config=_mapping(
            [("num_layers", 2), ("species_vocabulary", [6, 41])],
            reverse=reverse,
        ),
        model_floating_dtype="float64",
        model_state=_mapping(
            [
                ("private.raw.weight", torch.tensor([1234567.5], dtype=torch.float64)),
                ("private.buffer", torch.zeros((2, 3), dtype=torch.float64)),
            ],
            reverse=reverse,
        ),
        provenance=_mapping(
            [("purpose", "unit-test"), ("labels", ["z", "a"])],
            reverse=reverse,
        ),
        schema_version="reference_site_model_bundle_v1",
        species_vocabulary=(6, 41),
        template_bindings=tuple(bindings),
        version_metadata=_mapping(
            [("refsite_mlip_version", "0.1.0"), ("torch_version", "2.6.0")],
            reverse=reverse,
        ),
    )


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


def test_version_help_and_console_entry_configuration(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out == "refsite-mlip 0.1.0\n"

    parser = build_parser()
    assert parser.prog == "refsite-mlip"
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert 'refsite-mlip = "refsite_mlip.cli.main:main"' in pyproject.read_text()


def test_inspection_json_is_plain_stable_and_mapping_order_independent():
    first = summarize_bundle(_fake_bundle(reverse=False))
    second = summarize_bundle(_fake_bundle(reverse=True))
    _assert_plain(first)
    encoded = render_json(first)
    assert encoded == render_json(second)
    assert json.loads(encoded) == first
    assert "private.raw.weight" not in encoded
    assert "1234567.5" not in encoded
    assert first["template_ids"] == ["alpha", "zeta"]
    assert first["model"]["state"] == {
        "element_count": 7,
        "floating_dtype": "float64",
        "includes": ["parameters", "buffers"],
        "tensor_count": 2,
        "total_bytes": 56,
    }
    assert len(first["templates"]["alpha"]["phase_specification_fingerprint"]) == 64


def test_human_output_sorts_templates_and_spells_out_conventions():
    human = render_human(summarize_bundle(_fake_bundle(reverse=True)))
    assert human.index("  alpha\n") < human.index("  zeta\n")
    assert "Evaluation policy present: yes" in human
    assert "Evaluation policy present: no" in human
    assert "Stress sign: tensile_positive (no sign reversal)" in human
    assert 'Stress Voigt order: ["xx","yy","zz","yz","xz","xy"]' in human
    assert "Parameter/buffer state: 2 tensors, 7 elements, 56 bytes" in human
    assert "private.raw.weight" not in human


def test_cli_error_contract_and_concise_runtime_handling(monkeypatch, capsys):
    error = CLIError(
        "SAFE_LOAD_FAILURE",
        "safe bundle load or validation failed",
        stage="weights_only_load",
        bundle_path="bad\nmodel.pt",
        original_error=ValueError("unsafe payload"),
    )
    assert error.to_dict()["reason_code"] == "SAFE_LOAD_FAILURE"
    formatted = format_cli_error(error)
    assert "bad\\nmodel.pt" in formatted
    assert "weights_only_load" in formatted

    module = importlib.import_module("refsite_mlip.cli.inspect_bundle")

    def fail(path):
        del path
        raise error

    monkeypatch.setattr(module, "inspect_bundle", fail)
    assert main(["inspect-bundle", "bad-model.pt", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reason='SAFE_LOAD_FAILURE'" in captured.err
    assert "Traceback" not in captured.err

    with pytest.raises(CLIError):
        main(["inspect-bundle", "bad-model.pt", "--debug"])


def test_render_json_rejects_nonfinite_and_nonplain_values():
    with pytest.raises(ValueError, match="NaN or Infinity"):
        render_json({"value": float("nan")})
    with pytest.raises(TypeError, match="non-JSON metadata"):
        render_json({"value": torch.tensor(1.0)})


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect-bundle"],
        ["inspect-bundle", "model.pt", "--unknown"],
        ["unknown-command"],
    ],
)
def test_argparse_usage_errors_remain_exit_code_two(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
