from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import torch

from shapesplat.config import load_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.editing.metrics import compute_edit_metrics
from shapesplat.editing.ops import apply_edit
from shapesplat.editing.suite import run_edit_suite_for_scene, run_single_edit
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer


def _tmp_dir(name: str) -> Path:
    path = Path("outputs") / "test_editing_tmp" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tiny_cfg() -> dict:
    cfg = load_config("configs/minimal.yaml")
    cfg["image"]["size"] = 48
    cfg["training"]["visible_warmup_iters"] = 1
    cfg["training"]["hidden_prior_iters"] = 1
    cfg["training"]["joint_ownership_iters"] = 1
    cfg["training"]["edit_finetune_iters"] = 1
    cfg.setdefault("editing", {})["max_objects"] = 1
    return cfg


def _trained_scene():
    cfg = _tiny_cfg()
    image = make_synthetic_image(int(cfg["image"]["size"]))
    front = build_frontend(image, cfg)
    trainer = Trainer(front, cfg)
    trainer.train()
    return cfg, front, trainer


def test_edit_ops_do_not_modify_original() -> None:
    _, _, trainer = _trained_scene()
    original_means = trainer.scene.objects[0].means.detach().clone()
    original_opacity = trainer.scene.objects[0].opacity_logits.detach().clone()
    _ = apply_edit(trainer.scene, {"op": "remove", "object_id": 0})
    _ = apply_edit(trainer.scene, {"op": "translate", "object_id": 0, "translation": [0.1, 0.0, 0.0]})
    assert torch.allclose(trainer.scene.objects[0].means.detach(), original_means)
    assert torch.allclose(trainer.scene.objects[0].opacity_logits.detach(), original_opacity)


def test_run_single_edit() -> None:
    _, front, trainer = _trained_scene()
    out = _tmp_dir("single")
    metrics = run_single_edit(trainer.scene, trainer.renderer, front, 0, {"op": "remove", "object_id": 0}, out, save_visuals=True)
    assert "CollateralL1" in metrics
    assert "EditLocality" in metrics
    assert "DeletionResidual" in metrics
    assert (out / "metrics.json").exists()
    assert (out / "edited_render.png").exists()
    assert (out / "diff_heatmap.png").exists()


def test_run_edit_suite_for_scene() -> None:
    cfg, front, trainer = _trained_scene()
    out = _tmp_dir("suite")
    rows = run_edit_suite_for_scene(trainer.scene, trainer.renderer, front, out, object_ids=[0], edit_ops=["remove", "translate"], save_visuals=False, cfg=cfg)
    assert len(rows) == 2
    assert (out / "edit_metrics.json").exists()
    assert (out / "edit_summary.json").exists()


def test_edit_metrics_ranges() -> None:
    h = w = 16
    masks = torch.zeros(1, h, w)
    masks[0, 4:12, 4:12] = 1
    ownership = masks.clone()
    base = SimpleNamespace(rgb=torch.zeros(3, h, w), alpha=masks[0], ownership=ownership)
    edited = SimpleNamespace(rgb=torch.zeros(3, h, w), alpha=masks[0] * 0.25, ownership=ownership * 0.5)
    metrics = compute_edit_metrics(base, edited, masks, 0, "remove")
    assert 0.0 <= metrics["EditLocality"] <= 1.0
    assert metrics["DeletionResidual"] is not None
    assert torch.isfinite(torch.tensor(metrics["DeletionResidual"]))


def test_run_edit_demo_script_like() -> None:
    cfg, front, trainer = _trained_scene()
    out = _tmp_dir("demo_like")
    rows = run_edit_suite_for_scene(trainer.scene, trainer.renderer, front, out, object_ids=[0], edit_ops=["remove"], save_visuals=True, cfg=cfg)
    assert rows
    assert (out / "edit_summary.json").exists()
    assert (out / "object_000" / "remove" / "edit_triplet.png").exists()
