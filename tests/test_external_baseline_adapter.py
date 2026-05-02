from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.baselines.adapters import CommandTemplateAdapter
from shapesplat.baselines.export_inputs import export_baseline_inputs
from shapesplat.baselines.external_runner import run_external_baseline_dataset, run_external_baseline_for_image
from shapesplat.baselines.registry import list_adapters
from shapesplat.baselines.validate_outputs import validate_baseline_output_dir
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.example_dataset import create_example_dataset
from shapesplat.datasets.manifest import load_manifest
from shapesplat.frontend.file_mask_loader import load_mask_file


@pytest.fixture
def tmp_path(request) -> Path:
    root = ROOT / "outputs" / "test_external_baseline_adapter_tmp" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cfg(size: int = 32) -> dict:
    return merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": size}, "frontend": {"mask_source": "file"}})


def _make_input_spec(tmp_path: Path, image_id: str = "example_000"):
    manifest = create_example_dataset(tmp_path / "dataset", num_images=1, size=64)
    record = load_manifest(manifest)[0]
    cfg = _cfg(32)
    image = load_image(record.image_path, size=32)
    masks = load_mask_file(record.metadata["mask_path"], image_hw=image.shape[-2:], cfg=cfg).masks
    return export_baseline_inputs(image, masks, tmp_path / "inputs", image_id)


def test_list_adapters() -> None:
    adapters = list_adapters()
    assert "dummy_external" in adapters
    assert "command_template" in adapters


def test_validate_dummy_baseline_output(tmp_path: Path) -> None:
    spec = _make_input_spec(tmp_path)
    row = run_external_baseline_for_image("dummy_external", spec, tmp_path / "out", {"strict_validation": True})
    assert row["status"] == "success"
    result = validate_baseline_output_dir(tmp_path / "out", expected_num_objects=spec.num_objects, image_hw=(32, 32), strict=True)
    assert result["valid"] is True


def test_dummy_external_adapter_single_image(tmp_path: Path) -> None:
    spec = _make_input_spec(tmp_path)
    out = tmp_path / "dummy_external"
    row = run_external_baseline_for_image("dummy_external", spec, out, {"strict_validation": True})
    assert row["status"] == "success"
    assert (out / "render.png").exists()
    assert (out / "alpha.png").exists()
    assert (out / "ownership.npy").exists()
    assert (out / "metrics.json").exists()


def test_command_template_dry_run(tmp_path: Path) -> None:
    spec = _make_input_spec(tmp_path)
    adapter = CommandTemplateAdapter()
    cfg = {
        "name": "dry_method",
        "command": "python external_methods/example.py --image {image} --masks {masks} --out {output_dir}",
    }
    out = tmp_path / "dry_run"
    result = adapter.run(spec, out, cfg, dry_run=True)
    assert result.metadata["dry_run"] is True
    assert (out / "logs" / "command.txt").exists()


def test_run_external_baseline_dataset_dummy(tmp_path: Path) -> None:
    manifest = create_example_dataset(tmp_path / "dataset", num_images=2, size=64)
    cfg = _cfg(32)
    specs = []
    for record in load_manifest(manifest):
        image = load_image(record.image_path, size=32)
        masks = load_mask_file(record.metadata["mask_path"], image_hw=image.shape[-2:], cfg=cfg).masks
        specs.append(export_baseline_inputs(image, masks, tmp_path / record.image_id / "inputs", record.image_id))
    rows = run_external_baseline_dataset("dummy_external", specs, tmp_path / "dataset_out", {"strict_validation": True})
    assert len(rows) == 2
    assert (tmp_path / "dataset_out" / "external_baseline_summary.json").exists()

