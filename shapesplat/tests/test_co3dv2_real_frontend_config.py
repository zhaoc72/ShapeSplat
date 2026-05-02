from __future__ import annotations

import sys
from pathlib import Path

import torch
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.experiments.co3dv2_real_frontend import apply_dinov3_cli_overrides, best_iou_and_coverage, check_checkpoint_path
from shapesplat.frontend.dinov3_dependency_check import check_dinov3_dependencies
from shapesplat.frontend.dinov3_real import SUPPORTED_DINOV3_MODELS
from shapesplat.cache.validate_cache import validate_frontend_cache_manifest


@pytest.fixture
def local_tmp(request):
    root = ROOT / "outputs" / "test_co3dv2_real_frontend_config_tmp" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_local(path: str) -> dict:
    return load_config(path, runtime_overrides={"device": "cpu", "require_cuda_for_experiments": False, "allow_cpu_fallback": True})


def test_co3dv2_real_frontend_configs_load():
    debug = _load_local("configs/co3dv2_real_frontend_debug.yaml")
    main = _load_local("configs/co3dv2_real_frontend.yaml")
    assert debug["frontend"]["mask_source"] == "file"
    assert main["frontend"]["mask_source"] == "file"
    assert debug["frontend"]["dino_model_name"] == "dinov3_vits16"
    assert main["frontend"]["dino_model_name"] == "dinov3_vitl16"
    assert debug["frontend"]["dino_checkpoint"]
    assert main["frontend"]["dino_checkpoint"]
    assert debug["frontend"]["sam3_checkpoint"].endswith("sam3.pt")


def test_dinov3_checkpoint_missing_logic(local_tmp):
    missing = local_tmp / "missing.pth"
    report = check_checkpoint_path(missing, allow_missing=True)
    assert report["status"] == "missing"
    assert report["exists"] is False


def test_dinov3_dependency_check():
    report = check_dinov3_dependencies()
    assert isinstance(report, dict)
    assert "missing_required" in report
    assert "install_command" in report
    if report["missing_required"]:
        assert "torchmetrics" in report["install_command"]


def test_dino_model_name_mapping():
    assert "dinov3_vits16" in SUPPORTED_DINOV3_MODELS
    assert "dinov3_vitl16" in SUPPORTED_DINOV3_MODELS


def test_sam3_diagnostic_iou_logic():
    co3d = torch.zeros((1, 16, 16))
    co3d[:, 4:12, 4:12] = 1
    sam = torch.zeros((2, 16, 16))
    sam[0, 4:12, 4:12] = 1
    sam[1, 0:4, 0:4] = 1
    stats = best_iou_and_coverage(co3d, sam)
    assert stats["best_iou"] == 1.0
    assert stats["coverage"] == 1.0
    assert stats["num_sam_masks"] == 2


def test_cache_co3dv2_real_frontend_config_override():
    cfg = _load_local("configs/co3dv2_real_frontend_debug.yaml")
    apply_dinov3_cli_overrides(cfg, checkpoint="X:/fake/dino.pth", model_name="dinov3_vitl16", device="cpu")
    assert cfg["frontend"]["dino_checkpoint"] == "X:/fake/dino.pth"
    assert cfg["frontend"]["dino_model_name"] == "dinov3_vitl16"
    assert cfg["frontend"]["dino_device"] == "cpu"


def test_cache_validation_zero_records_warning(local_tmp):
    manifest = local_tmp / "empty_cache_manifest.csv"
    manifest.write_text("image_id,image_path,cache_dir,masks_path,descriptors_path,depth_path,meta_path,status,num_masks,descriptor_dim,warnings\n", encoding="utf-8")
    report = validate_frontend_cache_manifest(manifest)
    assert not report["valid"]
    assert report["num_valid"] == 0
    assert any("zero valid records" in w for w in report["warnings"])
