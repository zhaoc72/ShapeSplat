from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from shapesplat.cache.frontend_cache import load_frontend_output, save_frontend_output
from shapesplat.config import load_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.shape_prior.descriptor_precompute import point_statistics_descriptor, precompute_shape_descriptors
from shapesplat.shape_prior.retrieval import retrieve_shapes
from shapesplat.shape_prior.shape_bank_backend import build_shape_bank
from shapesplat.shape_prior.toy_shape_bank import ToyShapeBank

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path(request):
    """使用项目 outputs 下的测试目录，避开 Windows 临时目录权限问题。"""
    root = ROOT / "outputs" / "test_real_backend_pack_tmp" / f"{request.node.name}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cfg() -> dict:
    cfg = load_config("configs/minimal.yaml")
    cfg["image"]["size"] = 32
    cfg["frontend"]["sam_backend"] = "stub"
    cfg["frontend"]["dino_backend"] = "stub"
    cfg["frontend"]["depth_backend"] = "stub"
    cfg["training"]["visible_warmup_iters"] = 1
    cfg["training"]["hidden_prior_iters"] = 1
    cfg["training"]["joint_ownership_iters"] = 1
    cfg["training"]["edit_finetune_iters"] = 1
    return cfg


def test_frontend_cache_save_load(tmp_path: Path):
    cfg = _cfg()
    image = make_synthetic_image(32)
    front = build_frontend(image, cfg)
    save_frontend_output(front, tmp_path / "cache", image_id="synthetic", save_dino_features=True)
    cached = load_frontend_output(tmp_path / "cache", front.image)
    assert cached.masks.shape == front.masks.shape
    assert cached.descriptors.shape == front.descriptors.shape
    assert cached.depth.shape == front.depth.shape


def test_check_real_frontend_stub_mode():
    cfg = load_config("configs/local_real_frontend.yaml")
    cfg["frontend"]["sam_backend"] = "stub"
    cfg["frontend"]["dino_backend"] = "stub"
    cfg["frontend"]["depth_backend"] = "stub"
    image = make_synthetic_image(32)
    front = build_frontend(image, cfg)
    assert front.masks.shape[0] >= 1
    assert torch.isfinite(front.descriptors).all()
    assert torch.isfinite(front.depth).all()


def test_point_statistics_descriptor():
    points = torch.randn(128, 3)
    desc = point_statistics_descriptor(points, dim=16)
    assert desc.shape == (16,)
    assert torch.isfinite(desc).all()
    assert torch.isclose(torch.linalg.norm(desc), torch.tensor(1.0), atol=1e-5)


def test_precompute_shape_descriptors(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    np.save(src / "shape.npy", np.random.randn(32, 3).astype("float32"))
    rows = precompute_shape_descriptors(src, tmp_path / "out", descriptor_dim=16, mode="point_stats")
    assert len(rows) == 1
    assert (tmp_path / "out" / "shape.npz").exists()
    data = np.load(tmp_path / "out" / "shape.npz")
    assert data["descriptor"].shape == (16,)


def test_check_shape_retrieval_pipeline(tmp_path: Path):
    cfg = _cfg()
    cfg["frontend"]["dino_feature_dim"] = 16
    source = tmp_path / "toy"
    source.mkdir()
    toy = ToyShapeBank(descriptor_dim=16, device=torch.device("cpu"), points_per_shape=64)
    for asset in toy.assets:
        np.savez(source / f"{asset.shape_id}.npz", points=asset.points.numpy(), category=np.array(asset.category or "toy"))
    precompute_shape_descriptors(source, tmp_path / "prepared", descriptor_dim=16, mode="point_stats")
    cfg["shape_bank"]["backend"] = "file"
    cfg["shape_bank"]["root"] = str(tmp_path / "prepared")
    cfg["shape_bank"]["descriptor_dim"] = 16
    cfg["shape_bank"]["random_descriptor_if_missing"] = False
    image = make_synthetic_image(32)
    front = build_frontend(image, cfg)
    bank = build_shape_bank(cfg, descriptor_dim=front.descriptors.shape[1], device=front.descriptors.device)
    retrieved, weights, confidence = retrieve_shapes(front.descriptors, bank, top_k=3)
    assert len(retrieved) == front.descriptors.shape[0]
    assert weights.shape[0] == front.descriptors.shape[0]
    assert torch.isfinite(confidence).all()
