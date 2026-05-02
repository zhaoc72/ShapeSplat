from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from shapesplat.data.image_io import load_image
from shapesplat.baselines.protocol import read_baseline_output_spec


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        p = root / name
        if p.exists():
            return p
    return None


def _load_gray_png(path: Path) -> torch.Tensor:
    return load_image(path, size=None).mean(dim=0).clamp(0, 1)


def load_baseline_output(output_dir: str | Path, method_name: str, image_id: str) -> dict:
    """读取符合 baseline protocol 的输出。

    真实 baseline 后续只需要写出 render/alpha/ownership 或 object_i_alpha，
    就能通过该函数接入统一评估。
    """

    root = Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Baseline output directory not found: {root}")

    spec_path = root / "output_spec.json"
    spec = read_baseline_output_spec(spec_path) if spec_path.exists() else None

    render_path = Path(spec.render_path) if spec and spec.render_path else _first_existing(root, ["render_final.png", "render.png"])
    alpha_path = Path(spec.alpha_path) if spec and spec.alpha_path else _first_existing(root, ["alpha_final.png", "alpha.png"])
    ownership_path = Path(spec.ownership_path) if spec and spec.ownership_path else root / "ownership.npy"
    metrics_path = Path(spec.metrics_path) if spec and spec.metrics_path else root / "metrics.json"

    pred: dict = {"method_name": method_name, "image_id": image_id, "output_dir": str(root)}
    if render_path and render_path.exists():
        pred["rgb"] = load_image(render_path, size=None)
    if alpha_path and alpha_path.exists():
        pred["alpha"] = _load_gray_png(alpha_path)
    if ownership_path.exists():
        arr = np.load(ownership_path)
        pred["ownership"] = torch.from_numpy(arr).float()

    if "ownership" not in pred:
        alpha_files = sorted(root.glob("object_*_alpha.png"))
        if alpha_files:
            pred["ownership"] = torch.stack([_load_gray_png(p) for p in alpha_files], dim=0)
            pred["ownership"] = pred["ownership"] / pred["ownership"].sum(dim=0, keepdim=True).clamp_min(1e-6)
        else:
            raise FileNotFoundError(f"Missing ownership.npy and object_i_alpha.png in {root}")

    if "alpha" not in pred:
        pred["alpha"] = pred["ownership"].sum(dim=0).clamp(0, 1)
    if "rgb" not in pred:
        raise FileNotFoundError(f"Missing render.png/render_final.png in {root}")

    pred["bg_ownership"] = (1.0 - pred["alpha"]).clamp(0, 1)
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            pred["metrics"] = json.load(f)
    return pred

