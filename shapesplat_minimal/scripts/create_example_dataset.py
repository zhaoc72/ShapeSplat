from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def _draw_example(path: Path, size: int, idx: int) -> None:
    """绘制一张 synthetic-realistic RGB 示例图。

    这只是 batch runner 的 smoke-test dataset，不是真实 benchmark。
    """
    s = int(size)
    img = Image.new("RGB", (s, s), (238, 239, 236))
    draw = ImageDraw.Draw(img, "RGBA")
    shift = (idx * 7) % max(1, s // 5)
    colors = [
        (220, 70, 55, 255),
        (55, 130, 225, 235),
        (55, 175, 95, 245),
        (245, 190, 45, 230),
    ]

    for y in range(0, s, 10):
        shade = 232 + ((y + idx) // 10) % 6
        draw.line((0, y, s, y), fill=(shade, shade, shade, 45), width=1)

    n_obj = 2 + (idx % 3)
    if n_obj >= 1:
        draw.rounded_rectangle(
            (int(0.08 * s) + shift, int(0.16 * s), int(0.40 * s) + shift, int(0.50 * s)),
            radius=max(4, s // 18),
            fill=colors[0],
        )
    if n_obj >= 2:
        draw.ellipse(
            (int(0.42 * s), int(0.12 * s) + shift // 2, int(0.82 * s), int(0.52 * s) + shift // 2),
            fill=colors[1],
        )
    if n_obj >= 3:
        pts = [
            (int(0.20 * s), int(0.78 * s)),
            (int(0.48 * s) + shift // 3, int(0.45 * s)),
            (int(0.80 * s), int(0.84 * s)),
            (int(0.42 * s), int(0.94 * s)),
        ]
        draw.polygon(pts, fill=colors[2])
    if n_obj >= 4:
        draw.ellipse((int(0.36 * s), int(0.42 * s), int(0.64 * s), int(0.70 * s)), fill=colors[3])

    img = img.filter(ImageFilter.SMOOTH_MORE)
    img.save(path)


def create_example_dataset(out: str | Path, num_images: int = 4, size: int = 128) -> Path:
    """生成一个小型 manifest dataset，用于测试 batch experiment runner。"""
    out_dir = Path(out)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(int(num_images)):
        image_id = f"example_{i:03d}"
        image_path = image_dir / f"{image_id}.png"
        _draw_example(image_path, int(size), i)
        rows.append({"image_id": image_id, "image_path": f"images/{image_id}.png", "split": "test", "category": "toy"})

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "split", "category"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Example dataset saved to: {out_dir.resolve()}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny example dataset for ShapeSplat++ batch smoke tests.")
    parser.add_argument("--out", default="examples/example_dataset")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    create_example_dataset(args.out, args.num_images, args.size)


if __name__ == "__main__":
    main()
