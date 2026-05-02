from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

# Windows/Anaconda headless tests may not have a usable Tcl/Tk; Agg avoids GUI backend lookup.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch

from shapesplat.data.image_io import save_tensor_image
from shapesplat.renderer.types import RenderOutput


PALETTE = torch.tensor(
    [
        [0.90, 0.10, 0.12],
        [0.10, 0.45, 0.95],
        [0.15, 0.75, 0.30],
        [0.95, 0.65, 0.12],
        [0.65, 0.25, 0.95],
        [0.10, 0.75, 0.80],
    ],
    dtype=torch.float32,
)


def _colorize_label(label: torch.Tensor, num_labels: int) -> torch.Tensor:
    """把 [H,W] label map 转成 [3,H,W] 彩色图。"""
    h, w = label.shape
    out = torch.zeros(3, h, w, dtype=torch.float32)
    for i in range(num_labels):
        color = PALETTE[i % len(PALETTE)].view(3, 1, 1)
        out += color * (label == i).float()[None]
    return out.clamp(0, 1)


def save_mask_grid(masks: torch.Tensor, path: str | Path) -> None:
    """保存彩色 mask 网格。

    N=0 时直接抛出清晰错误，因为这通常意味着 front-end stub 没有产生可用前景。
    """
    if masks.shape[0] == 0:
        raise ValueError("save_mask_grid 收到 0 个 mask，无法保存 mask 可视化。")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = masks.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        color = PALETTE[i % len(PALETTE)].view(3, 1, 1)
        rgb = color * masks[i].detach().cpu().float()[None] + (1.0 - masks[i].detach().cpu().float()[None])
        ax.imshow(rgb.permute(1, 2, 0).numpy(), vmin=0, vmax=1)
        ax.set_title(f"mask {i}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_ownership_argmax(ownership: torch.Tensor, path: str | Path) -> None:
    """保存 ownership argmax 彩色图，用于检查真实图像上的 object attribution。"""
    if ownership.ndim != 3:
        raise ValueError(f"ownership 应为 [N,H,W]，实际形状: {tuple(ownership.shape)}")
    if ownership.shape[0] == 0:
        raise ValueError("save_ownership_argmax 收到 0 个 object。")
    arg = ownership.detach().cpu().argmax(dim=0)
    save_tensor_image(_colorize_label(arg, ownership.shape[0]), path)


def save_object_alphas(ownership: torch.Tensor, out_dir: str | Path) -> None:
    """逐物体保存 object_i_alpha.png。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(ownership.shape[0]):
        save_tensor_image(ownership[i], out_dir / f"object_{i}_alpha.png")


def save_render_outputs(render: RenderOutput, out_dir: str | Path) -> None:
    """保存 renderer 的核心输出图像。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tensor_image(render.rgb, out_dir / "render_final.png")
    save_tensor_image(render.alpha, out_dir / "alpha_final.png")
    save_ownership_argmax(render.ownership, out_dir / "ownership_argmax.png")
    save_object_alphas(render.ownership, out_dir)


def save_depth_map(depth: torch.Tensor, path: str | Path, normalize: bool = True) -> None:
    """保存 depth 可视化图。

    normalize=True 时仅用于可视化映射到 [0,1]，不表示 metric depth。
    """
    d = depth.detach().cpu().float()
    finite = torch.isfinite(d)
    if normalize:
        if bool(finite.any()):
            vals = d[finite]
            lo, hi = vals.min(), vals.max()
            d = (torch.where(finite, d, vals.median()) - lo) / (hi - lo).clamp_min(1e-6)
        else:
            d = torch.zeros_like(d)
    save_tensor_image(d.clamp(0, 1), path)


def save_input_with_mask_overlay(image: torch.Tensor, masks: torch.Tensor, path: str | Path) -> None:
    """把 masks 半透明叠加到输入图上，方便检查真实 RGB 输入的 stub mask 是否合理。"""
    if masks.shape[0] == 0:
        raise ValueError("save_input_with_mask_overlay 收到 0 个 mask。")
    base = image.detach().cpu().float().clamp(0, 1)
    overlay = base.clone()
    alpha = 0.45
    for i in range(masks.shape[0]):
        color = PALETTE[i % len(PALETTE)].view(3, 1, 1)
        m = masks[i].detach().cpu().float()[None]
        overlay = overlay * (1 - alpha * m) + color * (alpha * m)
    save_tensor_image(overlay, path)
