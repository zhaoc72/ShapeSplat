from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.datasets.benchmark.validator_v2 import validate_benchmark_v2
from shapesplat.datasets.converters.co3dv2_single import (
    CO3Dv2SingleConverter,
    convert_mask_to_stack,
    inspect_co3dv2_single,
    read_jgz_json,
)
from shapesplat.experiments.co3dv2_diagnostics import run_co3dv2_diagnostics


@pytest.fixture
def local_tmp(request):
    root = ROOT / "outputs" / "test_co3dv2_converter_tmp" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _make_fake_co3d(root: Path) -> Path:
    seq = root / "toy" / "seq001"
    for name in ["images", "masks", "depths", "depth_masks"]:
        (seq / name).mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 0] = 80 + idx * 20
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 10:26] = 255
        depth = np.full((32, 32), 128, dtype=np.uint8)
        Image.fromarray(rgb).save(seq / "images" / f"{idx:06d}.png")
        Image.fromarray(mask).save(seq / "masks" / f"{idx:06d}.png")
        Image.fromarray(depth).save(seq / "depths" / f"{idx:06d}.png")
        Image.fromarray(mask).save(seq / "depth_masks" / f"{idx:06d}.png")
    (seq / "pointcloud.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n", encoding="utf-8")
    return root


def test_inspect_fake_co3dv2_structure(local_tmp):
    root = _make_fake_co3d(local_tmp / "co3d")
    summary = inspect_co3dv2_single(root)
    assert summary["num_categories"] == 1
    assert summary["categories"][0]["sample_sequences"][0]["num_images"] == 2


def test_convert_fake_co3dv2_single(local_tmp):
    root = _make_fake_co3d(local_tmp / "co3d")
    manifest = CO3Dv2SingleConverter().convert(
        root,
        local_tmp / "bench",
        {"copy_files": True, "overwrite": True, "max_frames_per_sequence": 2},
    )
    assert manifest.exists()
    assert (local_tmp / "bench" / "masks").exists()
    report = validate_benchmark_v2(manifest, check_optional_gt=False)
    assert report["valid"]
    assert report["num_rows"] == 2


def test_mask_conversion_stack(local_tmp):
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[2:8, 3:9] = 255
    src = local_tmp / "mask.png"
    out = local_tmp / "mask.npy"
    Image.fromarray(mask).save(src)
    convert_mask_to_stack(src, out)
    arr = np.load(out)
    assert arr.shape == (1, 16, 20)
    assert arr.sum() > 0


def test_jgz_reader(local_tmp):
    path = local_tmp / "frame_annotations.jgz"
    payload = [{"sequence_name": "seq001"}]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    assert read_jgz_json(path)[0]["sequence_name"] == "seq001"


def test_run_co3dv2_diagnostics_small_fake(local_tmp):
    root = _make_fake_co3d(local_tmp / "co3d")
    manifest = CO3Dv2SingleConverter().convert(
        root,
        local_tmp / "bench",
        {"copy_files": True, "overwrite": True, "max_frames_per_sequence": 1},
    )
    result = run_co3dv2_diagnostics(manifest, "configs/final_ours.yaml", local_tmp / "diag", max_images=1)
    assert result["num_rows"] == 1
    assert (local_tmp / "diag" / "ours" / "ours_summary.json").exists()
