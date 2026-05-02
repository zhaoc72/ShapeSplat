from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from shapesplat.utils.visualization import PALETTE


def _to_rgb_image(t: torch.Tensor | None, h: int, w: int) -> torch.Tensor:
    if t is None:
        return torch.full((3, h, w), 0.72)
    x = t.detach().cpu().float().clamp(0, 1)
    if x.ndim == 2:
        return x[None].repeat(3, 1, 1)
    if x.ndim == 3 and x.shape[0] == 3:
        return x
    return torch.full((3, h, w), 0.72)


def _ownership_argmax_rgb(ownership: torch.Tensor | None, h: int, w: int) -> torch.Tensor:
    if ownership is None or ownership.ndim != 3 or ownership.shape[0] == 0:
        return torch.full((3, h, w), 0.72)
    label = ownership.detach().cpu().argmax(dim=0)
    out = torch.zeros(3, h, w)
    for i in range(ownership.shape[0]):
        out += PALETTE[i % len(PALETTE)].view(3, 1, 1) * (label == i).float()[None]
    return out.clamp(0, 1)


def _overlay(image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    base = image.detach().cpu().float().clamp(0, 1).clone()
    for i in range(masks.shape[0]):
        color = PALETTE[i % len(PALETTE)].view(3, 1, 1)
        m = masks[i].detach().cpu().float()[None]
        base = base * (1 - 0.45 * m) + color * (0.45 * m)
    return base.clamp(0, 1)


def make_comparison_grid(
    image: torch.Tensor,
    masks: torch.Tensor,
    method_outputs: dict[str, dict],
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """生成单图 qualitative comparison grid。

    该图用于快速检查 object ownership、foreground leakage 和 visual reconstruction。
    缺失的 method 输出会显示灰色 placeholder，避免单个方法失败影响整张图保存。
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, h, w = image.shape
    methods = list(method_outputs.keys())
    cols = 1 + len(methods)
    row_names = ["input+masks", "render", "alpha", "ownership"]
    fig, axes = plt.subplots(4, cols, figsize=(3.0 * cols, 8.5))
    if title:
        fig.suptitle(title)

    first_col = [_overlay(image, masks), _overlay(image, masks), masks.amax(dim=0)[None].repeat(3, 1, 1), _ownership_argmax_rgb(masks, h, w)]
    for r in range(4):
        ax = axes[r, 0]
        ax.axis("off")
        ax.imshow(first_col[r].permute(1, 2, 0).numpy())
        ax.set_title(row_names[r] if r == 0 else "")
        ax.set_ylabel(row_names[r])

    for c, method in enumerate(methods, start=1):
        pred = method_outputs.get(method, {})
        imgs = [
            _to_rgb_image(pred.get("rgb"), h, w),
            _to_rgb_image(pred.get("rgb"), h, w),
            _to_rgb_image(pred.get("alpha"), h, w),
            _ownership_argmax_rgb(pred.get("ownership"), h, w),
        ]
        for r, img in enumerate(imgs):
            ax = axes[r, c]
            ax.axis("off")
            ax.imshow(img.permute(1, 2, 0).numpy())
            if r == 0:
                ax.set_title(method)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_qualitative_index(comparison_root: str | Path, out_path: str | Path) -> None:
    """生成 Markdown index，方便快速浏览所有 qualitative_grid.png。"""

    root = Path(comparison_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grids = sorted(root.glob("per_image/*/qualitative_grid.png"))
    lines = ["# Qualitative Comparison Index", ""]
    for grid in grids:
        image_id = grid.parent.name
        rel = grid.relative_to(root).as_posix()
        lines.append(f"## {image_id}")
        lines.append(f"![{image_id}]({rel})")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

