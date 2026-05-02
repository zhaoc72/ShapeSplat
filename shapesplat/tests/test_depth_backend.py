from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.depth_backend import build_depth_backend
from shapesplat.frontend.depth_normalization import normalize_depth_to_canonical
from shapesplat.frontend.depth_stub import DepthStub
from shapesplat.frontend.pipeline import build_frontend


EPS = 1e-5


def _cfg(overrides=None):
    cfg = merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": 32}})
    if overrides:
        cfg = merge_config(cfg, overrides)
    cfg["device"] = "cpu"
    return cfg


def test_build_stub_depth_backend():
    cfg = _cfg({"frontend": {"depth_backend": "stub"}})
    backend = build_depth_backend(cfg)
    assert hasattr(backend, "predict_depth")


def test_stub_depth_outputs_valid_map():
    cfg = _cfg({"frontend": {"depth_backend": "stub"}})
    depth = build_depth_backend(cfg).predict_depth(make_synthetic_image(32))
    assert depth.shape == (32, 32)
    assert torch.isfinite(depth).all()
    assert float(depth.min()) >= cfg["camera"]["z_near"] - EPS
    assert float(depth.max()) <= cfg["camera"]["z_far"] + EPS


def test_auto_depth_fallback_without_checkpoint():
    cfg = _cfg({"frontend": {"depth_backend": "auto", "depth_checkpoint": "not_exist_depth_checkpoint.pt"}})
    backend = build_depth_backend(cfg)
    assert isinstance(backend, DepthStub)
    assert backend.predict_depth(make_synthetic_image(32)).shape == (32, 32)


def test_depth_normalization():
    cfg = _cfg()
    raw = torch.rand(32, 32) * 10 + 0.1
    norm = normalize_depth_to_canonical(raw, None, cfg, cfg["camera"]["z_near"], cfg["camera"]["z_far"])
    assert torch.isfinite(norm).all()
    assert float(norm.min()) >= cfg["camera"]["z_near"] - EPS
    assert float(norm.max()) <= cfg["camera"]["z_far"] + EPS


def test_frontend_uses_normalized_depth():
    cfg = _cfg()
    front = build_frontend(make_synthetic_image(32), cfg)
    assert front.depth.shape == (32, 32)
    assert torch.isfinite(front.depth).all()
    assert float(front.depth.min()) >= cfg["camera"]["z_near"] - EPS
    assert float(front.depth.max()) <= cfg["camera"]["z_far"] + EPS
