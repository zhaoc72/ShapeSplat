from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapesplat.config import load_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.renderer.backend import build_renderer
from shapesplat.renderer.camera_conversion import make_real_renderer_camera
from shapesplat.renderer.contract import validate_render_output
from shapesplat.renderer.gaussian_conversion import extract_gaussian_tensors
from shapesplat.renderer.real_renderer_check import check_real_renderer
from shapesplat.renderer.real_3dgs_adapter import Real3DGSRendererAdapter


@pytest.fixture
def tmp_path(request):
    """测试输出放到项目 outputs 中，避免 Windows 临时路径权限差异。"""

    root = ROOT / "outputs" / "test_real_3dgs_renderer_adapter_tmp" / f"{request.node.name}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cfg() -> dict:
    cfg = load_config("configs/real_3dgs_renderer.yaml")
    cfg["device"] = "cpu"
    cfg["image"]["size"] = 32
    cfg["renderer"]["backend"] = "soft"
    cfg["training"]["visible_warmup_iters"] = 1
    cfg["training"]["hidden_prior_iters"] = 1
    cfg["training"]["joint_ownership_iters"] = 1
    cfg["training"]["edit_finetune_iters"] = 1
    return cfg


def _front_scene(cfg: dict):
    image = make_synthetic_image(int(cfg["image"]["size"]))
    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    return front, scene


def test_gaussian_conversion():
    cfg = _cfg()
    front, scene = _front_scene(cfg)
    tensors = extract_gaussian_tensors(scene)
    for key in ["means", "scales", "rotations", "opacities", "colors", "object_ids", "branch_ids"]:
        assert key in tensors
    assert tensors["means"].shape[1] == 3
    assert tensors["scales"].shape == tensors["means"].shape
    assert tensors["rotations"].shape[1] == 4
    assert tensors["object_ids"].numel() == tensors["means"].shape[0]
    assert front.masks.shape[0] == len(scene.objects)


def test_camera_conversion():
    cfg = _cfg()
    front, _scene = _front_scene(cfg)
    params = make_real_renderer_camera(front.camera, cfg)
    assert params["tan_fovx"] > 0
    assert params["tan_fovy"] > 0
    assert params["view_matrix"].shape == (4, 4)
    assert params["projection_matrix"].shape == (4, 4)


def test_real_3dgs_adapter_unavailable():
    cfg = _cfg()
    cfg["renderer"]["backend"] = "real"
    cfg["renderer"]["real_3dgs"]["library"] = "auto"
    front, _scene = _front_scene(cfg)
    adapter = Real3DGSRendererAdapter(front.camera, cfg)
    assert hasattr(adapter, "available")
    assert adapter.available in {True, False}
    if not adapter.available:
        assert adapter.error_message


def test_renderer_auto_fallback_to_soft():
    cfg = _cfg()
    cfg["renderer"]["backend"] = "auto"
    cfg["renderer"]["fallback_to_soft"] = True
    front, scene = _front_scene(cfg)
    renderer = build_renderer(front.camera, cfg)
    render = renderer(scene)
    report = validate_render_output(render, len(scene.objects), front.camera.height, front.camera.width, strict=True)
    assert report["valid"] is True


def test_check_real_renderer_soft_fallback(tmp_path: Path):
    stats = check_real_renderer("configs/real_3dgs_renderer.yaml", tmp_path, backend="auto", allow_fallback=True, save_visuals=True)
    assert stats["contract"]["valid"] is True
    assert (tmp_path / "renderer_contract.json").exists()
    assert (tmp_path / "render.png").exists()
