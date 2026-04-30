from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from shapesplat.data.image_io import save_tensor_image
from shapesplat.evaluation.edit_metrics import compute_edit_metrics
from shapesplat.evaluation.metrics import compute_basic_metrics
from shapesplat.evaluation.report import merge_metrics, save_metrics_json
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.utils.logging import save_json
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid, save_render_outputs


def run_single_image_experiment(
    image: torch.Tensor,
    cfg: dict,
    out_dir: str | Path,
    image_id: str = "image",
    record=None,
    save_visuals: bool = True,
    save_checkpoint: bool = True,
    eval_metrics: bool = True,
) -> dict:
    """运行单张图像的最小 ShapeSplat++ 实验。

    这是单图流程的轻量封装：front-end、Gaussian 初始化、训练、渲染、保存和
    metrics 计算都在这里完成。batch runner 会逐图调用它，run_minimal.py 也可复用它。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save_visuals:
        save_tensor_image(image, out_path / "input.png")

    # batch experiment 中 record.metadata 可以携带 mask_path，从而启用 same-mask protocol。
    front = build_frontend(image, cfg, record=record)
    if front.masks.shape[0] == 0:
        raise RuntimeError(f"{image_id}: front-end produced no masks.")
    if save_visuals:
        save_mask_grid(front.masks, out_path / "masks.png")
        save_input_with_mask_overlay(front.image, front.masks, out_path / "input_mask_overlay.png")

    trainer = Trainer(front, cfg)
    loss_log = trainer.train()
    render = trainer.render()

    if save_visuals:
        save_render_outputs(render, out_path)
    # 保存原始 ownership tensor，便于 baseline protocol / comparison runner 统一读取。
    np.save(out_path / "ownership.npy", render.ownership.detach().cpu().float().numpy().astype("float32"))
    save_json(loss_log, out_path / "loss_log.json")
    if save_checkpoint:
        trainer.save_checkpoint(out_path / "checkpoint_minimal.pt")

    row: dict = {
        "image_id": image_id,
        "status": "success",
        "num_masks": int(front.masks.shape[0]),
        "num_objects": int(len(trainer.scene.objects)),
        "output_dir": str(out_path),
    }
    if eval_metrics:
        metrics = merge_metrics(
            compute_basic_metrics(render, front.masks),
            compute_edit_metrics(trainer.scene, trainer.renderer, front, render, cfg, object_id=0),
        )
        row.update(metrics)
        save_metrics_json(row, out_path / "metrics.json")
    return row
