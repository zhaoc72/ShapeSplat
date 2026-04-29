from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_ablation import run_ablation_suite
from shapesplat.config import DEFAULT_CONFIG, merge_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.optimization.losses import compute_losses
from shapesplat.renderer.soft_renderer import SoftGaussianRenderer
from shapesplat.utils.config_override import apply_overrides, load_ablation_file


def test_apply_overrides():
    cfg = {"ablation": {"use_hidden_prior": True}}
    new_cfg = apply_overrides(cfg, {"ablation.use_hidden_prior": False})
    assert cfg["ablation"]["use_hidden_prior"] is True
    assert new_cfg["ablation"]["use_hidden_prior"] is False


def test_ablation_config_loads():
    exps = load_ablation_file(ROOT / "configs" / "ablations.yaml")
    names = {e["name"] for e in exps}
    assert "full" in names
    assert "no_hidden_prior" in names


def test_losses_respect_ablation_switches():
    cfg = merge_config(
        DEFAULT_CONFIG,
        {
            "device": "cpu",
            "image": {"size": 32},
            "gaussians": {"visible_min": 8, "visible_max": 16, "hidden_base": 4},
            "ablation": {"use_hidden_prior": False, "use_ownership_loss": False},
        },
    )
    cfg["device"] = "cpu"
    front = build_frontend(make_synthetic_image(32), cfg)
    scene = initialize_scene(front, cfg)
    renderer = SoftGaussianRenderer(front.camera)
    render = renderer(scene)
    _, terms = compute_losses(scene, renderer, render, front, cfg, stage="joint")
    assert abs(terms["hidden_prior"]) < 1e-8
    assert abs(terms["identity"]) < 1e-8


def test_run_ablation_two_experiments():
    out = ROOT / "outputs" / "test_ablations_tmp"
    rows = run_ablation_suite(
        ROOT / "configs" / "minimal.yaml",
        ROOT / "configs" / "ablations.yaml",
        input_path=None,
        out=out,
        skip_existing=False,
        max_experiments=2,
    )
    summary = out / "ablation_summary.json"
    assert summary.exists()
    with open(summary, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(rows) >= 2
    assert len(data) >= 2
