from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import pytest
from PIL import Image

from create_example_image import create_example_image
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.image_io import load_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.optimization.losses import compute_losses
from shapesplat.renderer.soft_renderer import SoftGaussianRenderer


def _cfg(size: int = 64):
    cfg = merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": size}})
    cfg["device"] = "cpu"
    return cfg


@pytest.fixture
def tmp_path(request):
    """项目内 tmp_path fixture。

    某些 Windows 环境的系统临时目录没有列目录权限；这里仍然让测试使用 tmp_path，
    但把它映射到项目 outputs 下，保证 `pytest tests/test_real_input.py -v` 稳定。
    """
    root = ROOT / "outputs" / "test_real_input_tmp" / request.node.name
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_create_example_image(tmp_path):
    path = tmp_path / "create_example.png"
    create_example_image(path, size=64)
    assert path.exists()
    img = Image.open(path).convert("RGB")
    assert img.size == (64, 64)


def test_load_real_image(tmp_path):
    path = tmp_path / "load_real.png"
    create_example_image(path, size=80)
    img = load_image(path, size=64)
    assert img.shape == (3, 64, 64)
    assert 0 <= float(img.min()) <= float(img.max()) <= 1


def test_frontend_on_real_image(tmp_path):
    path = tmp_path / "frontend_real.png"
    create_example_image(path, size=64)
    cfg = _cfg(64)
    image = load_image(path, size=64)
    front = build_frontend(image, cfg)
    assert front.masks.shape[0] >= 1
    assert front.descriptors.shape[0] == front.masks.shape[0]


def test_real_input_one_step(tmp_path):
    path = tmp_path / "one_step_real.png"
    create_example_image(path, size=64)
    cfg = _cfg(64)
    image = load_image(path, size=64)
    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    renderer = SoftGaussianRenderer(front.camera)
    render = renderer(scene)
    loss, _ = compute_losses(scene, renderer, render, front, cfg, stage="visible")
    loss.backward()
    ok = False
    for p in scene.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0:
            ok = True
            break
    assert ok
