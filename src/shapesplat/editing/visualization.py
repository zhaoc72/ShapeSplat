from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch

from shapesplat.data.image_io import save_tensor_image
from shapesplat.editing.masks import make_edit_region, make_multi_object_non_edit_region


def save_diff_heatmap(base_rgb: torch.Tensor, edited_rgb: torch.Tensor, path: str | Path) -> torch.Tensor:
    """保存 RGB absolute difference heatmap，用于定性展示编辑影响范围。"""

    diff = (base_rgb.detach().cpu().float() - edited_rgb.detach().cpu().float()).abs().mean(dim=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3, 3))
    plt.axis("off")
    plt.imshow(diff.numpy())
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=120)
    plt.close()
    return diff


def save_edit_masks(edit_region: torch.Tensor, non_edit_region: torch.Tensor, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_tensor_image(edit_region.float(), out / "edit_region.png")
    save_tensor_image(non_edit_region.float(), out / "non_edit_region.png")


def save_edit_triplet(base_rgb: torch.Tensor, edited_rgb: torch.Tensor, diff: torch.Tensor, out_path: str | Path) -> None:
    """保存 original | edited | difference 三联图。"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    imgs = [base_rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1), edited_rgb.detach().cpu().permute(1, 2, 0).clamp(0, 1), diff.detach().cpu()]
    titles = ["original", "edited", "difference"]
    for ax, img, title in zip(axes, imgs, titles):
        ax.axis("off")
        ax.set_title(title)
        ax.imshow(img.numpy())
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_edit_visuals(base_render, edited_render, masks: torch.Tensor, object_id: int, out_dir: str | Path, op_name: str) -> None:
    """保存论文定性展示需要的 before/after/diff 和区域 mask。"""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oid = min(int(object_id), masks.shape[0] - 1, base_render.ownership.shape[0] - 1)
    masks = masks.to(base_render.rgb.device).float()
    edit_region = make_edit_region(masks[oid], base_render.ownership[oid].detach())
    non_edit = make_multi_object_non_edit_region(masks, base_render.ownership.detach(), oid)
    save_tensor_image(base_render.rgb, out / "original_render.png")
    save_tensor_image(edited_render.rgb, out / "edited_render.png")
    save_tensor_image(base_render.alpha, out / "original_alpha.png")
    save_tensor_image(edited_render.alpha, out / "edited_alpha.png")
    save_tensor_image(base_render.ownership[oid], out / "object_alpha_before.png")
    if oid < edited_render.ownership.shape[0]:
        save_tensor_image(edited_render.ownership[oid], out / "object_alpha_after.png")
    diff = save_diff_heatmap(base_render.rgb, edited_render.rgb, out / "diff_heatmap.png")
    save_edit_masks(edit_region, non_edit, out)
    save_edit_triplet(base_render.rgb, edited_render.rgb, diff, out / "edit_triplet.png")

