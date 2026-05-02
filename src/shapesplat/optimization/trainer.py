from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from shapesplat.frontend.pipeline import FrontEndOutput
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.renderer.backend import build_renderer
from shapesplat.renderer.types import RenderOutput
from .losses import compute_losses
from .edit_ops import edit_consistency_loss


class Trainer:
    """ShapeSplat++ 最小 trainer。

    分阶段训练的原因：hidden prior 太早加入会污染 visible branch；
    edit loss 太早加入会让 ownership 尚未稳定时出现不必要震荡。
    """

    def __init__(self, front: FrontEndOutput, cfg: Dict[str, Any]):
        self.front = front
        self.cfg = cfg
        self.scene = initialize_scene(front, cfg)
        self.renderer = build_renderer(front.camera, cfg)
        self._check_device_consistency()
        self.optim = torch.optim.Adam(self.scene.parameters(), lr=cfg["training"]["lr"])
        self.loss_log: List[Dict[str, float | str | int]] = []
        self.global_step = 0

    def _check_device_consistency(self) -> None:
        """检查 frontend / scene / camera 是否在同一设备。

        中文注释：Windows GPU 实验中 CPU/CUDA 张量混用会产生难读错误，所以在训练开始前主动失败。
        """
        expected = self.front.image.device
        if self.front.masks.device != expected or self.front.descriptors.device != expected or self.front.depth.device != expected:
            raise RuntimeError(
                "Frontend tensors are on inconsistent devices: "
                f"image={self.front.image.device}, masks={self.front.masks.device}, "
                f"descriptors={self.front.descriptors.device}, depth={self.front.depth.device}"
            )
        if self.front.camera.device != expected:
            raise RuntimeError(f"Camera device mismatch: camera={self.front.camera.device}, image={expected}")
        for name, param in self.scene.named_parameters():
            if param.device != expected:
                raise RuntimeError(f"Scene parameter device mismatch: {name} on {param.device}, expected {expected}")

    def render(self) -> RenderOutput:
        return self.renderer(self.scene)

    def _step(self, stage: str, it: int) -> None:
        self.optim.zero_grad(set_to_none=True)
        render = self.render()
        loss, terms = compute_losses(self.scene, self.renderer, render, self.front, self.cfg, stage)
        if stage == "edit":
            if self.cfg.get("ablation", {}).get("use_edit_consistency", True):
                e_rgb, e_alpha = edit_consistency_loss(self.scene, self.renderer, self.front, render, self.cfg)
            else:
                e_rgb = torch.tensor(0.0, device=self.front.image.device)
                e_alpha = torch.tensor(0.0, device=self.front.image.device)
            loss = loss + self.cfg["loss_weights"]["edit"] * e_rgb + self.cfg["loss_weights"]["edit_alpha"] * e_alpha
            terms["edit"] = float(e_rgb.detach().cpu())
            terms["edit_alpha"] = float(e_alpha.detach().cpu())
            terms["total"] = float(loss.detach().cpu())
        loss.backward()
        clip_grad_norm_(self.scene.parameters(), max_norm=1.0)
        self.optim.step()
        row = {"global_step": self.global_step, "stage": stage, "iter": it, "ablation_name": self.cfg.get("ablation_name", "full"), **terms}
        self.loss_log.append(row)
        self.global_step += 1
        if it % self.cfg["training"]["log_every"] == 0:
            print(f"[{stage} {it:03d}] loss={row['total']:.4f}")

    def train(self) -> List[Dict[str, float | str | int]]:
        schedule = [
            ("visible", self.cfg["training"]["visible_warmup_iters"]),
            ("hidden", self.cfg["training"]["hidden_prior_iters"]),
            ("joint", self.cfg["training"]["joint_ownership_iters"]),
            ("edit", self.cfg["training"]["edit_finetune_iters"]),
        ]
        for stage, n_iter in schedule:
            for i in tqdm(range(int(n_iter)), desc=stage):
                self._step(stage, i)
        return self.loss_log

    def save_checkpoint(self, path: str | Path) -> None:
        """保存最小 checkpoint，包含 scene 参数和 loss log。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"scene": self.scene.state_dict(), "loss_log": self.loss_log, "cfg": self.cfg}, path)
