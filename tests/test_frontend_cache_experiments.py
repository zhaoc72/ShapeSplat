from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapesplat.cache.attach import attach_cache_to_dataset
from shapesplat.cache.frontend_cache import (
    FrontendCacheManifestRecord,
    load_frontend_cache_manifest,
    save_frontend_output,
    write_frontend_cache_manifest,
)
from shapesplat.cache.validate_cache import validate_frontend_cache_dir
from shapesplat.cache.same_mask_export import cache_to_same_mask_dataset
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.datasets.example_dataset import create_example_dataset
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.experiments.batch_runner import run_batch_experiment
from shapesplat.frontend.pipeline import build_frontend


@pytest.fixture
def tmp_path(request):
    """把测试临时文件放到项目 outputs 下，避免 Windows 沙箱路径差异影响文件读写。"""

    root = ROOT / "outputs" / "test_frontend_cache_experiments_tmp" / f"{request.node.name}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fast_cfg(size: int = 32) -> dict:
    cfg = merge_config(
        DEFAULT_CONFIG,
        {
            "device": "cpu",
            "image": {"size": size},
            "gaussians": {"visible_min": 8, "visible_max": 16, "hidden_base": 4},
            "frontend": {"sam_backend": "stub", "dino_backend": "stub", "depth_backend": "stub"},
            "training": {
                "visible_warmup_iters": 1,
                "hidden_prior_iters": 1,
                "joint_ownership_iters": 1,
                "edit_finetune_iters": 1,
            },
        },
    )
    return cfg


def _make_cached_dataset(tmp_path: Path, num_images: int = 2):
    """创建一个小数据集并为每张图保存 frontend cache。"""

    dataset_dir = tmp_path / "dataset"
    manifest = create_example_dataset(dataset_dir, num_images=num_images, size=64)
    cfg = _fast_cfg(32)
    dataset = build_dataset_from_manifest(manifest, image_size=32)
    cache_root = tmp_path / "frontend_cache"
    records: list[FrontendCacheManifestRecord] = []
    for item in dataset:
        cache_dir = cache_root / item["image_id"]
        front = build_frontend(item["image"], cfg, record=item["record"])
        rec = save_frontend_output(front, cache_dir, item["image_id"])
        records.append(
            FrontendCacheManifestRecord(
                image_id=item["image_id"],
                image_path=str(item["record"].image_path),
                cache_dir=str(cache_dir),
                masks_path=rec.masks_path,
                descriptors_path=rec.descriptors_path,
                depth_path=rec.depth_path,
                meta_path=rec.meta_path,
                num_masks=int(front.masks.shape[0]),
                descriptor_dim=int(front.descriptors.shape[1]),
            )
        )
    cache_manifest = cache_root / "cache_manifest.csv"
    write_frontend_cache_manifest(records, cache_manifest)
    return manifest, cache_manifest, cache_root, cfg


def test_frontend_cache_manifest(tmp_path: Path):
    manifest, cache_manifest, _cache_root, _cfg = _make_cached_dataset(tmp_path, num_images=1)
    loaded = load_frontend_cache_manifest(cache_manifest)
    assert "example_000" in loaded
    assert Path(loaded["example_000"].cache_dir).exists()
    assert Path(manifest).exists()


def test_validate_frontend_cache_dir(tmp_path: Path):
    _manifest, cache_manifest, _cache_root, _cfg = _make_cached_dataset(tmp_path, num_images=1)
    rec = load_frontend_cache_manifest(cache_manifest)["example_000"]
    report = validate_frontend_cache_dir(rec.cache_dir, image_hw=(32, 32))
    assert report["valid"] is True
    assert report["num_masks"] >= 1


def test_build_frontend_use_cache(tmp_path: Path):
    manifest, cache_manifest, _cache_root, cfg = _make_cached_dataset(tmp_path, num_images=1)
    dataset = build_dataset_from_manifest(manifest, image_size=32)
    attach_cache_to_dataset(dataset, cache_manifest=cache_manifest)
    item = dataset[0]
    cfg["frontend_cache"] = {"use_cache": True, "fallback_to_compute": False, "validate_on_load": True}
    front = build_frontend(item["image"], cfg, record=item["record"], use_cache=True)
    assert front.masks.shape[-2:] == (32, 32)
    assert front.descriptors.shape[0] == front.masks.shape[0]
    assert front.depth.shape == (32, 32)


def test_cache_to_same_mask_dataset(tmp_path: Path):
    manifest, cache_manifest, _cache_root, _cfg = _make_cached_dataset(tmp_path, num_images=1)
    out_manifest = cache_to_same_mask_dataset(manifest, cache_manifest, tmp_path / "cached_same_mask", copy_images=True, overwrite=True)
    assert out_manifest.exists()
    assert (out_manifest.parent / "masks" / "example_000.npy").exists()


def test_batch_runner_with_cache(tmp_path: Path):
    manifest, cache_manifest, _cache_root, cfg = _make_cached_dataset(tmp_path, num_images=2)
    dataset = build_dataset_from_manifest(manifest, image_size=32)
    attach_cache_to_dataset(dataset, cache_manifest=cache_manifest)
    cfg["frontend_cache"] = {"use_cache": True, "fallback_to_compute": False, "validate_on_load": True}
    rows = run_batch_experiment(dataset, cfg, tmp_path / "batch", max_images=2, save_checkpoint=False, use_frontend_cache=True)
    assert len(rows) == 2
    assert all(row["status"] == "success" for row in rows)
    assert (tmp_path / "batch" / "example_000" / "metrics.json").exists()
