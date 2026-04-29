from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch

from shapesplat.data.image_io import save_tensor_image
from shapesplat.renderer.soft_renderer import RenderOutput


def save_mask_grid(masks: torch.Tensor, path: str | Path) -> None:
    """保存 mask 网格；用于检查 SAM3 retained visible masks 是否合理。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, masks.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < masks.shape[0]:
            ax.imshow(masks[i].detach().cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"mask {i}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_ownership_argmax(ownership: torch.Tensor, path: str | Path) -> None:
    """保存 ownership argmax，可用于检查 object attribution 是否串台。"""
    if ownership.shape[0] == 0:
        return
    arg = ownership.detach().cpu().argmax(dim=0).float()
    if ownership.shape[0] > 1:
        arg = arg / float(ownership.shape[0] - 1)
    save_tensor_image(arg, path)


def save_object_alphas(ownership: torch.Tensor, out_dir: str | Path) -> None:
    """逐物体保存 ownership/alpha-like map。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(ownership.shape[0]):
        save_tensor_image(ownership[i], out_dir / f"object_{i}_alpha.png")


def save_render_outputs(render: RenderOutput, out_dir: str | Path) -> None:
    """保存 renderer 的关键输出图像。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tensor_image(render.rgb, out_dir / "render_final.png")
    save_tensor_image(render.alpha, out_dir / "alpha_final.png")
    save_ownership_argmax(render.ownership, out_dir / "ownership_argmax.png")
    save_object_alphas(render.ownership, out_dir)
