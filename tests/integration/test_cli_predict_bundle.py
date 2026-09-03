from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

import refsite_mlip.cli.predict as predict_module
from refsite_mlip.cli.errors import CLIError
from refsite_mlip.cli.main import main
from refsite_mlip.cli.predict import ExtXYZPredictionConfig, predict_extxyz
from refsite_mlip.data import StructureSample
from refsite_mlip.inference import load_reference_site_predictor
from refsite_mlip.models import (
    capture_reference_site_model_bundle,
    save_reference_site_model_bundle,
)
from refsite_mlip.transport import EVAL_ADAPTIVE, TRAIN_FIXED

from test_cli_inspect_bundle import _typed_crystal_data
from test_model_bundle_runtime import _capture_case


@pytest.fixture(scope="module")
def prediction_bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cli-predict-bundle")
    values = _capture_case(_typed_crystal_data())
    _, model, _, samples, _, _, policies, mixed = values
    mixed_path = directory / "mixed.pt"
    save_reference_site_model_bundle(mixed_path, mixed)

    bindings = {value.template_id: value for value in mixed.template_bindings}
    no_policy = capture_reference_site_model_bundle(
        model=model,
        structural_artifacts={
            key: value.structural_artifact for key, value in bindings.items()
        },
        phase_specifications={
            key: value.phase_specification for key, value in bindings.items()
        },
        evaluation_policies=None,
        default_template_id=mixed.default_template_id,
        provenance={"purpose": "cli-predict-no-policy"},
    )
    no_policy_path = directory / "no-policy.pt"
    save_reference_site_model_bundle(no_policy_path, no_policy)
    return {
        "mixed": mixed_path,
        "no_policy": no_policy_path,
        "samples": samples,
        "policies": policies,
    }


def _atoms(sample, index: int, *, labeled: bool, template_info: bool = True):
    info = {"source_metadata": f"frame-{index}"}
    if template_info:
        info["template"] = sample.template_id
    atoms = Atoms(
        numbers=sample.atomic_numbers.detach().cpu().numpy(),
        positions=sample.positions.detach().cpu().numpy(),
        cell=sample.cell.detach().cpu().numpy(),
        pbc=sample.pbc.detach().cpu().numpy(),
        info=info,
    )
    atoms.arrays["input_order"] = np.arange(len(atoms), dtype=np.int64) + 100 * index
    if labeled:
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=1000.0 + index,
            free_energy=900.0 + index,
            forces=np.full((len(atoms), 3), 70.0 + index),
            stress=np.arange(6, dtype=float) + index,
        )
    return atoms


def _write_input(path: Path, samples, *, labeled: bool, template_info: bool = True):
    frames = tuple(
        _atoms(sample, index, labeled=labeled, template_info=template_info)
        for index, sample in enumerate(samples)
    )
    write(path, frames, format="extxyz")
    return frames


def _read_all(path: Path):
    values = read(path, index=":", format="extxyz")
    return tuple(values if isinstance(values, list) else [values])


def _geometry_samples(frames, template_ids):
    return tuple(
        StructureSample(
            sample_id=f"predict:{index:06d}",
            positions=torch.tensor(atoms.get_positions(), dtype=torch.float64),
            atomic_numbers=torch.tensor(
                atoms.get_atomic_numbers(), dtype=torch.long
            ),
            cell=torch.tensor(atoms.cell.array, dtype=torch.float64),
            pbc=torch.tensor(atoms.get_pbc(), dtype=torch.bool),
            origin=torch.zeros(3, dtype=torch.float64),
            template_id=template_ids[index],
        )
        for index, atoms in enumerate(frames)
    )


def _assert_output_matches_direct(input_frames, output_frames, direct, *, tolerance=6e-8):
    assert len(input_frames) == len(output_frames) == len(direct.structures)
    for source, output, prediction in zip(
        input_frames, output_frames, direct.structures
    ):
        np.testing.assert_array_equal(output.numbers, source.numbers)
        np.testing.assert_allclose(output.positions, source.positions, atol=5e-9, rtol=0.0)
        np.testing.assert_allclose(output.cell.array, source.cell.array, atol=5e-9, rtol=0.0)
        np.testing.assert_array_equal(output.pbc, source.pbc)
        np.testing.assert_array_equal(output.arrays["input_order"], source.arrays["input_order"])
        assert output.info["source_metadata"] == source.info["source_metadata"]
        assert output.info["refsite_template_id"] == prediction.template_id
        assert output.info["refsite_solver_path"] in (TRAIN_FIXED, EVAL_ADAPTIVE)
        assert len(output.info["refsite_bundle_sha256"]) == 64
        assert output.get_potential_energy() == pytest.approx(
            float(prediction.energy), abs=tolerance, rel=tolerance
        )
        assert output.get_potential_energy(force_consistent=True) == pytest.approx(
            float(prediction.energy), abs=tolerance, rel=tolerance
        )
        np.testing.assert_allclose(
            output.get_forces(), prediction.forces.numpy(), atol=tolerance, rtol=tolerance
        )
        np.testing.assert_allclose(
            output.get_stress(),
            prediction.stress_voigt.numpy(),
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_allclose(
            output.get_stress(voigt=False),
            prediction.stress.numpy(),
            atol=tolerance,
            rtol=tolerance,
        )
        tensor = output.get_stress(voigt=False)
        voigt = output.get_stress()
        assert voigt[3] == pytest.approx(tensor[1, 2])
        assert voigt[4] == pytest.approx(tensor[0, 2])
        assert voigt[5] == pytest.approx(tensor[0, 1])


def test_train_fixed_single_multi_default_index_and_property_selection(
    prediction_bundle, tmp_path
):
    zeta = (prediction_bundle["samples"][0], prediction_bundle["samples"][2])
    input_path = tmp_path / "zeta.xyz"
    _write_input(input_path, zeta, labeled=False, template_info=False)

    output_path = tmp_path / "default.xyz"
    report = predict_extxyz(
        ExtXYZPredictionConfig(
            bundle_path=prediction_bundle["mixed"],
            input_path=input_path,
            output_path=output_path,
            properties=("energy",),
            batch_size=1,
        )
    )
    assert report["frame_count"] == 2
    assert report["template_frame_counts"] == {"zeta": 2}
    assert report["forces"] is None and report["stress"] is None
    for frame in _read_all(output_path):
        assert set(frame.calc.results) == {"energy", "free_energy"}
    with pytest.raises(CLIError) as caught:
        predict_extxyz(
            ExtXYZPredictionConfig(
                bundle_path=prediction_bundle["mixed"],
                input_path=input_path,
                output_path=output_path,
            )
        )
    assert caught.value.reason_code == "OUTPUT_EXISTS"

    selected = tmp_path / "selected.xyz"
    selection = predict_extxyz(
        ExtXYZPredictionConfig(
            bundle_path=prediction_bundle["mixed"],
            input_path=input_path,
            output_path=selected,
            index="1",
            template_id="zeta",
        )
    )
    assert selection["frame_count"] == 1
    chosen = _read_all(selected)[0]
    np.testing.assert_allclose(chosen.positions, _read_all(input_path)[1].positions)
    assert "forces" in chosen.calc.results and "stress" not in chosen.calc.results


def test_mixed_template_key_adaptive_direct_parity_order_stress_and_labels(
    prediction_bundle, tmp_path
):
    input_path = tmp_path / "mixed-labeled.xyz"
    _write_input(input_path, prediction_bundle["samples"], labeled=True)
    input_bytes = input_path.read_bytes()
    bundle_bytes = prediction_bundle["mixed"].read_bytes()
    parsed_input = _read_all(input_path)
    input_geometry = tuple(
        (
            frame.positions.copy(),
            frame.numbers.copy(),
            frame.cell.array.copy(),
            frame.pbc.copy(),
        )
        for frame in parsed_input
    )

    output_path = tmp_path / "mixed-adaptive.xyz"
    report = predict_extxyz(
        ExtXYZPredictionConfig(
            bundle_path=prediction_bundle["mixed"],
            input_path=input_path,
            output_path=output_path,
            template_key="template",
            solver_path="eval-adaptive",
            properties=("energy", "forces", "stress"),
            dtype="float64",
            batch_size=2,
        )
    )
    predictor = load_reference_site_predictor(
        prediction_bundle["mixed"], device="cpu", dtype=torch.float64
    )
    direct = predictor.predict_samples(
        _geometry_samples(
            parsed_input,
            tuple(frame.info["template"] for frame in parsed_input),
        ),
        solver_path=EVAL_ADAPTIVE,
        compute_forces=True,
        compute_stress=True,
    )
    output = _read_all(output_path)
    _assert_output_matches_direct(parsed_input, output, direct)
    assert report["bundle_sha256"] == predictor.bundle_fingerprint
    assert all(
        frame.info["refsite_bundle_sha256"] == predictor.bundle_fingerprint
        for frame in output
    )
    assert report["template_frame_counts"] == {"alpha": 1, "zeta": 2}
    assert report["stress"]["sign"] == "tensile_positive"
    assert report["stress"]["voigt_order"] == ["xx", "yy", "zz", "yz", "xz", "xy"]
    assert input_path.read_bytes() == input_bytes
    assert prediction_bundle["mixed"].read_bytes() == bundle_bytes
    for frame, state in zip(parsed_input, input_geometry):
        np.testing.assert_array_equal(frame.positions, state[0])
        np.testing.assert_array_equal(frame.numbers, state[1])
        np.testing.assert_array_equal(frame.cell.array, state[2])
        np.testing.assert_array_equal(frame.pbc, state[3])


def test_label_independence_and_batch_size_parity(prediction_bundle, tmp_path):
    labeled = tmp_path / "labeled.xyz"
    unlabeled = tmp_path / "unlabeled.xyz"
    _write_input(labeled, prediction_bundle["samples"], labeled=True)
    _write_input(unlabeled, prediction_bundle["samples"], labeled=False)
    outputs = []
    for source, batch_size, name in (
        (labeled, 1, "labeled-out.xyz"),
        (unlabeled, 16, "unlabeled-out.xyz"),
    ):
        target = tmp_path / name
        predict_extxyz(
            ExtXYZPredictionConfig(
                bundle_path=prediction_bundle["mixed"],
                input_path=source,
                output_path=target,
                template_key="template",
                properties=("energy", "forces", "stress"),
                batch_size=batch_size,
            )
        )
        outputs.append(_read_all(target))
    for left, right in zip(*outputs):
        assert left.get_potential_energy() == right.get_potential_energy()
        np.testing.assert_array_equal(left.get_forces(), right.get_forces())
        np.testing.assert_array_equal(left.get_stress(), right.get_stress())


def test_summary_human_json_determinism(prediction_bundle, tmp_path, capsys):
    input_path = tmp_path / "input.xyz"
    _write_input(input_path, prediction_bundle["samples"], labeled=False)
    output_path = tmp_path / "output.xyz"
    arguments = [
        "predict",
        "--bundle",
        str(prediction_bundle["mixed"]),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--template-key",
        "template",
        "--properties",
        "energy,forces,stress",
        "--json",
    ]
    assert main(arguments) == 0
    first = capsys.readouterr()
    assert first.err == ""
    parsed = json.loads(first.out)
    assert parsed["frame_count"] == 3
    assert parsed["output_path"] == str(output_path)
    assert parsed["requested_properties"] == ["energy", "forces", "stress"]
    assert parsed["energy"]["min"] <= parsed["energy"]["mean"] <= parsed["energy"]["max"]
    assert parsed["forces"]["component_rms"] >= 0.0
    assert parsed["forces"]["max_force_norm"] >= 0.0

    assert main([*arguments, "--overwrite"]) == 0
    second = capsys.readouterr()
    assert second.err == ""
    assert second.out == first.out

    human_output = tmp_path / "human.xyz"
    human_args = [value for value in arguments if value != "--json"]
    human_args[human_args.index(str(output_path))] = str(human_output)
    assert main(human_args) == 0
    human = capsys.readouterr()
    assert human.err == ""
    assert "Frames: 3" in human.out
    assert "Templates: alpha=1, zeta=2" in human.out
    assert "tensile-positive, xx yy zz yz xz xy" in human.out


def test_one_predictor_load_rng_bundle_input_and_atomic_failure(
    prediction_bundle, tmp_path, monkeypatch
):
    source = tmp_path / "input.xyz"
    _write_input(source, prediction_bundle["samples"][:1], labeled=True)
    target = tmp_path / "output.xyz"
    source_bytes = source.read_bytes()
    bundle_bytes = prediction_bundle["mixed"].read_bytes()
    rng = torch.get_rng_state().clone()
    calls = []
    original_load = predict_module.load_reference_site_predictor

    def counted_load(*args, **kwargs):
        calls.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(
        predict_module, "load_reference_site_predictor", counted_load
    )
    predict_extxyz(
        ExtXYZPredictionConfig(
            bundle_path=prediction_bundle["mixed"],
            input_path=source,
            output_path=target,
            template_id="zeta",
        )
    )
    assert len(calls) == 1
    assert torch.equal(torch.get_rng_state(), rng)
    assert source.read_bytes() == source_bytes
    assert prediction_bundle["mixed"].read_bytes() == bundle_bytes

    old_output = b"preserved existing output"
    target.write_bytes(old_output)
    original_replace = predict_module.os.replace

    def fail_replace(*args, **kwargs):
        del args, kwargs
        raise OSError("injected atomic commit failure")

    monkeypatch.setattr(predict_module.os, "replace", fail_replace)
    with pytest.raises(CLIError) as caught:
        predict_extxyz(
            ExtXYZPredictionConfig(
                bundle_path=prediction_bundle["mixed"],
                input_path=source,
                output_path=target,
                template_id="zeta",
                overwrite=True,
            )
        )
    monkeypatch.setattr(predict_module.os, "replace", original_replace)
    assert caught.value.reason_code == "OUTPUT_COMMIT_FAILED"
    assert target.read_bytes() == old_output
    assert not list(tmp_path.glob(".output.xyz.*.tmp"))


def test_output_collision_symlink_corrupt_inputs_and_structured_errors(
    prediction_bundle, tmp_path, capsys
):
    source = tmp_path / "input.xyz"
    _write_input(source, prediction_bundle["samples"][:1], labeled=False)
    base = [
        "predict",
        "--bundle",
        str(prediction_bundle["mixed"]),
        "--input",
        str(source),
        "--output",
        str(source),
        "--template-id",
        "zeta",
    ]
    assert main(base) == 1
    collision = capsys.readouterr()
    assert collision.out == ""
    assert "INPUT_OUTPUT_COLLISION" in collision.err

    bundle = prediction_bundle["mixed"]
    bundle_before = bundle.read_bytes()
    bundle_args = list(base)
    bundle_args[bundle_args.index(str(source), bundle_args.index("--output"))] = str(
        bundle
    )
    bundle_args.append("--overwrite")
    assert main(bundle_args) == 1
    bundle_collision = capsys.readouterr()
    assert bundle_collision.out == ""
    assert "BUNDLE_OUTPUT_COLLISION" in bundle_collision.err
    assert bundle.read_bytes() == bundle_before

    bundle_hardlink = tmp_path / "bundle-hardlink.pt"
    os.link(bundle, bundle_hardlink)
    hardlink_args = list(bundle_args)
    hardlink_args[
        hardlink_args.index(str(bundle), hardlink_args.index("--output"))
    ] = str(bundle_hardlink)
    assert main(hardlink_args) == 1
    hardlink_collision = capsys.readouterr()
    assert hardlink_collision.out == ""
    assert "BUNDLE_OUTPUT_COLLISION" in hardlink_collision.err
    assert bundle.read_bytes() == bundle_before

    existing = tmp_path / "existing.xyz"
    existing.write_text("keep")
    link = tmp_path / "linked.xyz"
    link.symlink_to(existing)
    link_args = list(base)
    link_args[link_args.index(str(source), link_args.index("--output"))] = str(link)
    assert main(link_args) == 1
    symlink = capsys.readouterr()
    assert symlink.out == ""
    assert "OUTPUT_SYMLINK_REJECTED" in symlink.err
    assert existing.read_text() == "keep"

    malformed = tmp_path / "malformed.xyz"
    malformed.write_text("this is not extxyz\n")
    malformed_output = tmp_path / "malformed-out.xyz"
    malformed_args = list(base)
    malformed_args[malformed_args.index(str(source), malformed_args.index("--input"))] = str(malformed)
    malformed_args[malformed_args.index(str(source), malformed_args.index("--output"))] = str(malformed_output)
    assert main(malformed_args) == 1
    malformed_result = capsys.readouterr()
    assert malformed_result.out == ""
    assert "MALFORMED_EXTXYZ" in malformed_result.err
    assert not malformed_output.exists()

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a portable bundle")
    corrupt_output = tmp_path / "corrupt-out.xyz"
    corrupt_args = list(base)
    corrupt_args[corrupt_args.index(str(prediction_bundle["mixed"]))] = str(corrupt)
    corrupt_args[corrupt_args.index(str(source), corrupt_args.index("--output"))] = str(corrupt_output)
    assert main(corrupt_args) == 1
    corrupt_result = capsys.readouterr()
    assert corrupt_result.out == ""
    assert "predictor_reason_code=" in corrupt_result.err
    assert str(corrupt) in corrupt_result.err
    assert not corrupt_output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("partial_pbc", "NONPERIODIC_STRUCTURE"),
        ("singular_cell", "SINGULAR_CELL"),
        ("unknown_template", "UNKNOWN_TEMPLATE"),
        ("unsupported_species", "UNSUPPORTED_SPECIES"),
        ("too_many_atoms", "INVALID_N_GT_M"),
        ("missing_template_key", "MISSING_TEMPLATE_KEY"),
    ],
)
def test_geometry_template_and_composition_fail_fast(
    prediction_bundle, tmp_path, mutation, expected_reason
):
    atoms = _atoms(prediction_bundle["samples"][0], 0, labeled=False)
    template_key = "template"
    if mutation == "partial_pbc":
        atoms.pbc = [True, False, True]
    elif mutation == "singular_cell":
        atoms.cell = np.zeros((3, 3))
    elif mutation == "unknown_template":
        atoms.info["template"] = "absent"
    elif mutation == "unsupported_species":
        atoms.numbers[0] = 8
    elif mutation == "too_many_atoms":
        atoms.append(Atom(6, position=[0.2, 0.4, 0.6]))
        atoms.append(Atom(41, position=[0.3, 0.5, 0.7]))
    elif mutation == "missing_template_key":
        atoms.info.pop("template")
    source = tmp_path / f"{mutation}.xyz"
    target = tmp_path / f"{mutation}-out.xyz"
    write(source, atoms, format="extxyz")
    with pytest.raises(CLIError) as caught:
        predict_extxyz(
            ExtXYZPredictionConfig(
                bundle_path=prediction_bundle["mixed"],
                input_path=source,
                output_path=target,
                template_key=template_key,
            )
        )
    assert caught.value.reason_code == expected_reason
    assert caught.value.frame_index == 0
    assert caught.value.sample_id == "predict:000000"
    assert caught.value.solver_path == TRAIN_FIXED
    assert not target.exists()


def test_empty_input_fails_without_output(prediction_bundle, tmp_path):
    source = tmp_path / "empty.xyz"
    source.write_bytes(b"")
    target = tmp_path / "empty-out.xyz"
    with pytest.raises(CLIError) as caught:
        predict_extxyz(
            ExtXYZPredictionConfig(
                bundle_path=prediction_bundle["mixed"],
                input_path=source,
                output_path=target,
                template_id="zeta",
            )
        )
    assert caught.value.reason_code == "EMPTY_INPUT"
    assert not target.exists()


def test_adaptive_missing_policy_and_unavailable_cuda_fail_fast(
    prediction_bundle, tmp_path
):
    source = tmp_path / "input.xyz"
    _write_input(source, prediction_bundle["samples"][:1], labeled=False)
    with pytest.raises(CLIError) as caught:
        predict_extxyz(
            ExtXYZPredictionConfig(
                bundle_path=prediction_bundle["no_policy"],
                input_path=source,
                output_path=tmp_path / "missing-policy.xyz",
                template_id="zeta",
                solver_path=EVAL_ADAPTIVE,
            )
        )
    assert caught.value.reason_code == "POLICY_CONTEXT_MISMATCH"
    assert caught.value.template_id == "zeta"
    assert caught.value.frame_index == 0

    with pytest.raises(CLIError) as caught:
        predict_extxyz(
            ExtXYZPredictionConfig(
                bundle_path=prediction_bundle["mixed"],
                input_path=source,
                output_path=tmp_path / "cuda.xyz",
                template_id="zeta",
                device="cuda:99999",
            )
        )
    assert caught.value.reason_code == "UNAVAILABLE_CUDA_DEVICE"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable on this node; retained for the Milestone 9D gate",
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("solver", [TRAIN_FIXED, EVAL_ADAPTIVE])
def test_cuda_focused_energy_force_stress_smoke(
    prediction_bundle, tmp_path, dtype, solver
):
    source = tmp_path / f"cuda-{dtype}-{solver}.xyz"
    target = tmp_path / f"cuda-{dtype}-{solver}-out.xyz"
    _write_input(source, prediction_bundle["samples"][:1], labeled=False)
    report = predict_extxyz(
        ExtXYZPredictionConfig(
            bundle_path=prediction_bundle["mixed"],
            input_path=source,
            output_path=target,
            template_id="zeta",
            solver_path=solver,
            properties=("energy", "forces", "stress"),
            device="cuda",
            dtype=dtype,
        )
    )
    assert report["dtype"] == dtype and report["device"] == "cuda"
    output = _read_all(target)[0]
    assert np.isfinite(output.get_potential_energy())
    assert np.all(np.isfinite(output.get_forces()))
    assert np.all(np.isfinite(output.get_stress()))
