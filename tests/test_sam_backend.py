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
from shapesplat.frontend.mask_postprocess import postprocess_masks
from shapesplat.frontend.sam3_stub import Sam3Stub
from shapesplat.frontend.sam_backend import build_sam_backend
from shapesplat.frontend.types import MaskSet


def _cfg(overrides=None):
    cfg = merge_config(DEFAULT_CONFIG, {"device": "cpu", "image": {"size": 32}})
    if overrides:
        cfg = merge_config(cfg, overrides)
    cfg["device"] = "cpu"
    return cfg


def test_build_stub_backend():
    cfg = _cfg({"frontend": {"sam_backend": "stub"}})
    backend = build_sam_backend(cfg)
    assert hasattr(backend, "predict_masks")


def test_stub_backend_outputs_maskset():
    cfg = _cfg({"frontend": {"sam_backend": "stub"}})
    backend = build_sam_backend(cfg)
    out = backend.predict_masks(make_synthetic_image(32))
    assert out.masks.ndim == 3
    assert out.confidences.shape == (out.masks.shape[0],)
    assert out.boxes.shape == (out.masks.shape[0], 4)
    assert out.masks.dtype == torch.float32


def test_auto_backend_fallback_without_checkpoint():
    cfg = _cfg({"frontend": {"sam_backend": "auto", "sam3_checkpoint": "not_exist_sam3_checkpoint.pt"}})
    backend = build_sam_backend(cfg)
    assert isinstance(backend, Sam3Stub)
    out = backend.predict_masks(make_synthetic_image(32))
    assert out.masks.shape[0] >= 1


def test_mask_postprocess_empty_fallback():
    empty = MaskSet(
        masks=torch.zeros((0, 32, 32)),
        confidences=torch.zeros((0,)),
        boxes=torch.zeros((0, 4)),
    )
    cfg = _cfg()
    out = postprocess_masks(empty, cfg, (32, 32))
    assert out.masks.shape[0] == 1
    assert out.boxes.shape == (1, 4)
