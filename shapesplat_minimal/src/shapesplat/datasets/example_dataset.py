from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


PALETTE = [(220, 70, 55), (55, 130, 225), (55, 175, 95), (245, 190, 45)]


def _shape_specs(size: int, idx: int) -> list[tuple[str, object]]:
    """为第 idx 张 toy image 生成一组确定性的几何物体。

    这是 smoke test / same-mask protocol 使用的小型合成数据，不是真实 benchmark。
    """

    s = int(size)
    shift = (idx * 7) % max(1, s // 5)
    specs: list[tuple[str, object]] = [
        ("round_rect", (int(0.08 * s) + shift, int(0.16 * s), int(0.40 * s) + shift, int(0.50 * s))),
        ("ellipse", (int(0.42 * s), int(0.12 * s) + shift // 2, int(0.82 * s), int(0.52 * s) + shift // 2)),
        (
            "polygon",
            [
                (int(0.20 * s), int(0.78 * s)),
                (int(0.48 * s) + shift // 3, int(0.45 * s)),
                (int(0.80 * s), int(0.84 * s)),
                (int(0.42 * s), int(0.94 * s)),
            ],
        ),
        ("ellipse", (int(0.36 * s), int(0.42 * s), int(0.64 * s), int(0.70 * s))),
    ]
    return specs[: 2 + (idx % 3)]


def _draw_one_mask(size: int, spec: tuple[str, object]) -> Image.Image:
    """绘制单个物体的 amodal 几何 mask；后面会转成 visible mask。"""

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    kind, geom = spec
    if kind == "round_rect":
        draw.rounded_rectangle(geom, radius=max(4, size // 18), fill=1)
    elif kind == "ellipse":
        draw.ellipse(geom, fill=1)
    elif kind == "polygon":
        draw.polygon(geom, fill=1)
    return mask


def _draw_example(image_path: Path, mask_path: Path, mask_png_path: Path, size: int, idx: int) -> None:
    """生成一张 RGB 图和对应的 retained visible instance masks。

    mask 是遮挡后的可见区域，用于 same-mask protocol；它不是 amodal mask。
    """

    s = int(size)
    img = Image.new("RGB", (s, s), (238, 239, 236))
    bg = ImageDraw.Draw(img, "RGBA")
    for y in range(0, s, 10):
        shade = 232 + ((y + idx) // 10) % 6
        bg.line((0, y, s, y), fill=(shade, shade, shade, 45), width=1)

    specs = _shape_specs(s, idx)
    raw_masks = [np.array(_draw_one_mask(s, spec), dtype=bool) for spec in specs]
    visible_masks = []
    occupied_later = np.zeros((s, s), dtype=bool)
    # 从后往前扣除被后绘制物体遮挡的区域，得到 visible instance masks。
    for raw in reversed(raw_masks):
        visible = raw & (~occupied_later)
        visible_masks.append(visible)
        occupied_later |= raw
    visible_masks = list(reversed(visible_masks))

    for i, visible in enumerate(visible_masks):
        overlay = Image.new("RGBA", (s, s), (*PALETTE[i % len(PALETTE)], 235))
        alpha = Image.fromarray((visible.astype(np.uint8) * 255), mode="L")
        img.paste(overlay, (0, 0), alpha)

    img = img.filter(ImageFilter.SMOOTH_MORE)
    img.save(image_path)

    stack = np.stack([m.astype(np.uint8) for m in visible_masks], axis=0)
    np.save(mask_path, stack)
    label = np.zeros((s, s), dtype=np.uint8)
    for i, m in enumerate(visible_masks, start=1):
        label[m] = i
    Image.fromarray(label, mode="L").save(mask_png_path)


def create_example_dataset(
    out_dir: str | Path,
    num_images: int = 4,
    size: int = 128,
) -> Path:
    """创建带 image、visible masks 和 manifest 的 toy dataset。

    输出用于 smoke test、dataset runner 和 same-mask protocol 检查，不代表真实
    benchmark。返回 manifest.csv 的路径，方便 tests 和脚本继续加载。
    """

    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    mask_png_dir = out_dir / "mask_png"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_png_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(int(num_images)):
        image_id = f"example_{i:03d}"
        image_path = image_dir / f"{image_id}.png"
        mask_path = mask_dir / f"{image_id}.npy"
        mask_png_path = mask_png_dir / f"{image_id}.png"
        _draw_example(image_path, mask_path, mask_png_path, int(size), i)
        rows.append(
            {
                "image_id": image_id,
                "image_path": f"images/{image_id}.png",
                "mask_path": f"masks/{image_id}.npy",
                "split": "test",
                "category": "toy",
            }
        )

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "mask_path", "split", "category"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest
