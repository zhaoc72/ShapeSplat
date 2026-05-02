from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shapesplat.cache.frontend_cache import save_frontend_output
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.evaluation.edit_metrics import compute_edit_metrics
from shapesplat.evaluation.metrics import compute_basic_metrics
from shapesplat.evaluation.report import merge_metrics, save_metrics_json
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.integration.capabilities import detect_all_capabilities
from shapesplat.integration.report import save_integration_report
from shapesplat.optimization.trainer import Trainer
from shapesplat.utils.logging import save_json
from shapesplat.utils.seed import seed_everything
from shapesplat.utils.visualization import save_depth_map, save_input_with_mask_overlay, save_mask_grid, save_render_outputs


def _frontend_stats(front) -> dict:
    desc = front.descriptors.detach().cpu().float()
    depth = front.depth.detach().cpu().float()
    return {
        "num_masks": int(front.masks.shape[0]),
        "masks_shape": list(front.masks.shape),
        "descriptors_shape": list(front.descriptors.shape),
        "descriptor_finite": bool(torch.isfinite(desc).all()),
        "descriptor_norm_mean": float(desc.norm(dim=1).mean()) if desc.numel() else 0.0,
        "depth_shape": list(front.depth.shape),
        "depth_finite": bool(torch.isfinite(depth).all()),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_mean": float(depth.mean()),
    }


def _save_descriptor_stats(front, out_dir: Path) -> dict:
    desc = front.descriptors.detach().cpu().float()
    stats = {
        "shape": list(desc.shape),
        "finite": bool(torch.isfinite(desc).all()),
        "norms": [float(x) for x in desc.norm(dim=1)] if desc.ndim == 2 else [],
    }
    (out_dir / "descriptor_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def run_real_integration_smoke(
    cfg: dict,
    image_path: str | Path | None,
    out_dir: str | Path,
    save_cache: bool = True,
    run_reconstruction: bool = True,
    force_stub_ok: bool = True,
) -> dict:
    """运行本地真实组件集成 smoke test。

    该流程用于验证 backend 配置、auto fallback、frontend cache 和
    reconstruction plumbing；它不代表最终论文实验质量。
    """

    out = Path(out_dir)
    frontend_dir = out / "frontend"
    recon_dir = out / "reconstruction"
    cache_dir = out / "cache"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.get("seed", 7)))

    capabilities = detect_all_capabilities(cfg)
    warnings = list(capabilities.get("overall", {}).get("warnings", []))
    if not force_stub_ok:
        real_requested = any(
            capabilities[k].get("requested") == "real" and not capabilities[k].get("available")
            for k in ["sam", "dino", "depth", "renderer"]
        )
        if real_requested:
            report = {"status": "failed", "capabilities": capabilities, "warnings": warnings, "error": "real backend requested but unavailable"}
            save_integration_report(report, out)
            return report

    if image_path:
        image = load_image(image_path, size=int(cfg.get("image", {}).get("size", 128)))
    else:
        image = make_synthetic_image(int(cfg.get("image", {}).get("size", 128)))
    save_tensor_image(image, frontend_dir / "input.png")

    try:
        front = build_frontend(image, cfg, cache_dir=cache_dir if save_cache else None, save_cache=save_cache)
        if front.masks.shape[0] == 0:
            raise RuntimeError("frontend produced zero masks")
        save_mask_grid(front.masks, frontend_dir / "masks.png")
        save_input_with_mask_overlay(front.image, front.masks, frontend_dir / "input_mask_overlay.png")
        save_depth_map(front.depth, frontend_dir / "depth.png")
        stats = _frontend_stats(front)
        desc_stats = _save_descriptor_stats(front, frontend_dir)
        save_json(stats, frontend_dir / "frontend_stats.json")
    except Exception as exc:
        report = {"status": "failed", "capabilities": capabilities, "warnings": warnings, "error": str(exc)}
        save_integration_report(report, out)
        return report

    metrics: dict = {}
    if run_reconstruction:
        try:
            recon_dir.mkdir(parents=True, exist_ok=True)
            trainer = Trainer(front, cfg)
            loss_log = trainer.train()
            render = trainer.render()
            save_render_outputs(render, recon_dir)
            np.save(recon_dir / "ownership.npy", render.ownership.detach().cpu().float().numpy().astype("float32"))
            save_json(loss_log, recon_dir / "loss_log.json")
            metrics = merge_metrics(
                compute_basic_metrics(render, front.masks),
                compute_edit_metrics(trainer.scene, trainer.renderer, front, render, cfg, object_id=0),
            )
            metrics.update({"status": "success", "output_dir": str(recon_dir)})
            save_metrics_json(metrics, recon_dir / "metrics.json")
        except Exception as exc:
            warnings.append(f"reconstruction failed: {exc}")

    status = "success_with_fallback" if any(capabilities[k].get("will_fallback") for k in ["sam", "dino", "depth", "shape_bank", "renderer"]) else "success"
    report = {
        "status": status,
        "capabilities": capabilities,
        "frontend_stats": stats,
        "descriptor_stats": desc_stats,
        "metrics": metrics,
        "cache_paths": {"cache_dir": str(cache_dir)} if save_cache else {},
        "warnings": warnings,
    }
    save_integration_report(report, out)
    return report
