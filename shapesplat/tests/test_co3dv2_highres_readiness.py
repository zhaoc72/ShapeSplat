from __future__ import annotations

import csv
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image

from shapesplat.experiments.co3dv2_highres_readiness import check_co3dv2_highres_ready, save_co3dv2_highres_readiness_report


def _work_dir(name: str) -> Path:
    root = Path("outputs") / "test_co3dv2_highres_readiness" / f"{name}_{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fake_manifest(root: Path) -> Path:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((479, 640, 3), dtype=np.uint8)).save(image_dir / "000.png")
    np.save(mask_dir / "000.npy", np.ones((1, 479, 640), dtype="float32"))
    manifest = root / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "mask_path", "split", "subset"])
        writer.writeheader()
        writer.writerow({"image_id": "co3d_000", "image_path": "images/000.png", "mask_path": "masks/000.npy", "split": "test", "subset": "co3dv2_single"})
    return manifest


def _cfg(tmp_path: Path) -> dict:
    return {
        "runtime": {"device": "cpu", "allow_cpu_fallback": True, "require_cuda_for_experiments": False},
        "image": {"resize_mode": "keep_aspect", "long_side": 640, "size": 640},
        "frontend": {
            "mask_source": "file",
            "mask_resize_mode": "nearest",
            "dino_input_size": 448,
            "dino_checkpoint": str(tmp_path / "missing_dino.pth"),
        },
        "frontend_cache": {"cache_manifest": None},
        "renderer": {"backend": "soft", "fallback_to_soft": True},
        "shape_bank": {"backend": "toy", "root": None},
        "debug": {"allow_debug_iteration_cap": False},
    }


def test_check_co3dv2_highres_ready_debug_mode():
    tmp_path = _work_dir("debug")
    manifest = _fake_manifest(tmp_path)
    report = check_co3dv2_highres_ready(_cfg(tmp_path), manifest, strict=False)
    assert "warnings" in report
    assert "errors" in report
    assert "checks" in report
    assert any(c["name"] == "mask_resize_nearest" and c["ok"] for c in report["checks"])


def test_strict_flags_soft_and_toy():
    tmp_path = _work_dir("strict")
    manifest = _fake_manifest(tmp_path)
    ckpt = tmp_path / "dino.pth"
    ckpt.write_bytes(b"fake")
    cfg = _cfg(tmp_path)
    cfg["frontend"]["dino_checkpoint"] = str(ckpt)
    report = check_co3dv2_highres_ready(cfg, manifest, strict=True)
    assert report["ready"] is False
    assert any("SoftGaussianRenderer" in e or "ToyShapeBank" in e for e in report["errors"])


def test_save_readiness_report():
    tmp_path = _work_dir("save")
    report = {"ready": False, "strict": False, "errors": ["x"], "warnings": ["y"], "checks": [{"name": "a", "ok": True, "level": "info", "message": "m"}], "summary": {}}
    save_co3dv2_highres_readiness_report(report, tmp_path / "out")
    assert (tmp_path / "out" / "readiness.json").exists()
    assert (tmp_path / "out" / "readiness.md").exists()
