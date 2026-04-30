from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.renderer.backend import build_renderer
from shapesplat.renderer.real_renderer_placeholder import RealRendererPlaceholder
from shapesplat.renderer.soft_renderer import SoftGaussianRenderer


def _cfg(overrides=None):
    cfg = merge_config(
        DEFAULT_CONFIG,
        {
            "device": "cpu",
            "image": {"size": 32},
            "gaussians": {"visible_min": 8, "visible_max": 16, "hidden_base": 4},
        },
    )
    if overrides:
        cfg = merge_config(cfg, overrides)
    cfg["device"] = "cpu"
    return cfg


def _scene_and_render(cfg):
    front = build_frontend(make_synthetic_image(int(cfg["image"]["size"])), cfg)
    scene = initialize_scene(front, cfg)
    renderer = build_renderer(front.camera, cfg)
    render = renderer(scene)
    return front, scene, renderer, render


def test_build_soft_renderer_backend():
    cfg = _cfg({"renderer": {"backend": "soft"}})
    front = build_frontend(make_synthetic_image(32), cfg)
    renderer = build_renderer(front.camera, cfg)
    assert callable(renderer)
    assert isinstance(renderer, SoftGaussianRenderer)


def test_soft_renderer_output_contract():
    cfg = _cfg({"renderer": {"backend": "soft"}})
    front, scene, _, render = _scene_and_render(cfg)
    h = w = int(cfg["image"]["size"])
    assert render.rgb.shape == (3, h, w)
    assert render.alpha.shape == (h, w)
    assert render.depth.shape == (h, w)
    assert render.contributions.shape == (len(scene.objects), h, w)
    assert render.ownership.shape == (len(scene.objects), h, w)
    assert render.bg_ownership.shape == (h, w)
    assert render.contributions.shape[0] == front.masks.shape[0]


def test_ownership_normalization():
    cfg = _cfg({"renderer": {"backend": "soft"}})
    _, _, _, render = _scene_and_render(cfg)
    ownership_sum = render.bg_ownership + render.ownership.sum(dim=0)
    assert torch.isfinite(ownership_sum).all()
    assert float((ownership_sum - 1.0).abs().mean().detach().cpu()) < 1e-3


def test_auto_renderer_fallback():
    cfg = _cfg({"renderer": {"backend": "auto", "real_renderer_module": None, "fallback_to_soft": True}})
    front = build_frontend(make_synthetic_image(32), cfg)
    renderer = build_renderer(front.camera, cfg)
    assert isinstance(renderer, SoftGaussianRenderer)


def test_real_renderer_placeholder_raises():
    cfg = _cfg()
    front = build_frontend(make_synthetic_image(32), cfg)
    scene = initialize_scene(front, cfg)
    renderer = RealRendererPlaceholder(front.camera, cfg)
    with pytest.raises(NotImplementedError):
        renderer(scene)
