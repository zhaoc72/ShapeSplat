from __future__ import annotations

import csv
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.datasets.benchmark.builder_v2 import build_benchmark_from_existing_manifest
from shapesplat.datasets.benchmark.cache_binding import bind_frontend_cache_to_benchmark
from shapesplat.datasets.benchmark.manifest_v2 import load_benchmark_manifest
from shapesplat.datasets.benchmark.schema import REQUIRED_COLUMNS
from shapesplat.datasets.benchmark.summary_v2 import summarize_benchmark_v2
from shapesplat.datasets.benchmark.validator_v2 import validate_benchmark_v2
from shapesplat.datasets.converters.generic_folder import GenericFolderConverter
from shapesplat.datasets.converters.gso_template import GSOConverterTemplate
from shapesplat.datasets.example_dataset import create_example_dataset


@pytest.fixture
def tmp_path(request):
    """把临时 benchmark 放在 outputs 下，避免 Windows tmp 权限和路径差异。"""

    root = ROOT / "outputs" / "test_final_benchmark_pack_tmp" / f"{request.node.name}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_benchmark_schema_columns():
    assert {"image_id", "image_path", "mask_path", "split"}.issubset(set(REQUIRED_COLUMNS))


def test_build_benchmark_from_existing_manifest(tmp_path: Path):
    src = tmp_path / "example"
    manifest = create_example_dataset(src, num_images=2, size=32)
    out_manifest = build_benchmark_from_existing_manifest(manifest, tmp_path / "bench", source_dataset="example", overwrite=True)
    assert out_manifest.exists()
    report = validate_benchmark_v2(out_manifest)
    assert report["valid"] is True
    assert report["num_valid"] == 2


def test_generic_folder_converter(tmp_path: Path):
    src = tmp_path / "example"
    create_example_dataset(src, num_images=1, size=32)
    manifest = GenericFolderConverter().convert(src, tmp_path / "generic", {"source_dataset": "example", "overwrite": True})
    assert manifest.exists()
    assert validate_benchmark_v2(manifest)["valid"] is True


def test_benchmark_summary(tmp_path: Path):
    src = tmp_path / "example"
    manifest = create_example_dataset(src, num_images=2, size=32)
    out_manifest = build_benchmark_from_existing_manifest(manifest, tmp_path / "bench", source_dataset="example", overwrite=True)
    summary = summarize_benchmark_v2(out_manifest)
    assert summary["num_images"] == 2
    assert summary["source_dataset_counts"]["example"] == 2


def test_bind_frontend_cache_to_benchmark(tmp_path: Path):
    src = tmp_path / "example"
    manifest = create_example_dataset(src, num_images=1, size=32)
    bench_manifest = build_benchmark_from_existing_manifest(manifest, tmp_path / "bench", source_dataset="example", overwrite=True)
    cache_dir = tmp_path / "cache" / "example_000"
    cache_dir.mkdir(parents=True)
    np.save(cache_dir / "masks.npy", np.ones((1, 32, 32), dtype="float32"))
    np.save(cache_dir / "descriptors.npy", np.ones((1, 16), dtype="float32"))
    np.save(cache_dir / "depth.npy", np.ones((32, 32), dtype="float32"))
    (cache_dir / "frontend_meta.json").write_text("{}", encoding="utf-8")
    cache_manifest = tmp_path / "cache_manifest.csv"
    with open(cache_manifest, "w", encoding="utf-8", newline="") as f:
        fields = ["image_id", "image_path", "cache_dir", "masks_path", "descriptors_path", "depth_path", "meta_path", "status", "num_masks", "descriptor_dim", "warnings"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"image_id": "example_000", "image_path": "", "cache_dir": str(cache_dir), "masks_path": str(cache_dir / "masks.npy"), "descriptors_path": str(cache_dir / "descriptors.npy"), "depth_path": str(cache_dir / "depth.npy"), "meta_path": str(cache_dir / "frontend_meta.json"), "status": "valid", "num_masks": "1", "descriptor_dim": "16", "warnings": ""})
    out_manifest = bind_frontend_cache_to_benchmark(bench_manifest, cache_manifest, tmp_path / "bench" / "manifest_with_cache.csv")
    record = load_benchmark_manifest(out_manifest)[0]
    assert record.frontend_cache_dir is not None
    assert record.frontend_cache_status == "valid"


def test_template_converters_raise_clear_error(tmp_path: Path):
    src = tmp_path / "raw_gso"
    src.mkdir()
    with pytest.raises(NotImplementedError, match="template"):
        GSOConverterTemplate().convert(src, tmp_path / "out")
