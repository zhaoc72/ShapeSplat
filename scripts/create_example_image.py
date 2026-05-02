from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def create_example_image(out: str | Path, size: int = 128) -> Path:
    """生成 synthetic-realistic RGB 示例图。

    这张图只用于真实输入路径 smoke test：它包含浅色背景、多个彩色物体和轻微遮挡，
    不是训练数据，也不代表真实数据分布。
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = int(size)
    img = Image.new("RGB", (s, s), (238, 239, 235))
    draw = ImageDraw.Draw(img, "RGBA")

    # 轻微背景纹理，避免真实输入路径只适配纯白背景。
    for y in range(0, s, 8):
        shade = 232 + (y // 8) % 5
        draw.line((0, y, s, y), fill=(shade, shade, shade, 55), width=1)

    # 三个前景物体部分重叠，测试 mask/ownership 可视化是否稳定。
    draw.rounded_rectangle((int(0.12 * s), int(0.18 * s), int(0.58 * s), int(0.60 * s)), radius=max(4, s // 16), fill=(220, 70, 55, 255))
    draw.ellipse((int(0.42 * s), int(0.12 * s), int(0.86 * s), int(0.56 * s)), fill=(55, 130, 225, 235))
    pts = [
        (int(0.24 * s), int(0.78 * s)),
        (int(0.50 * s), int(0.45 * s)),
        (int(0.78 * s), int(0.86 * s)),
        (int(0.42 * s), int(0.94 * s)),
    ]
    draw.polygon(pts, fill=(55, 175, 95, 245))
    draw.ellipse((int(0.35 * s), int(0.42 * s), int(0.65 * s), int(0.72 * s)), fill=(245, 190, 45, 230))

    img = img.filter(ImageFilter.SMOOTH_MORE)
    img.save(out)
    print(f"Example image saved to: {out.resolve()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic-realistic RGB example image.")
    parser.add_argument("--out", default="examples/test_image.png", help="输出 PNG 路径")
    parser.add_argument("--size", type=int, default=128, help="图像边长")
    args = parser.parse_args()
    create_example_image(args.out, args.size)


if __name__ == "__main__":
    main()
