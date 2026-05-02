from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from create_example_image import create_example_image
from run_minimal import run_pipeline


def main() -> None:
    """真实输入路径 demo。

    注意：这个脚本仍然使用 Sam3Stub、DinoV3Stub、DepthStub 和 SoftGaussianRenderer，
    只用于验证真实 RGB 图像读取、resize、front-end stub、初始化、训练和保存逻辑。
    """
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal real-input demo.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="配置文件路径")
    parser.add_argument("--input", default="examples/test_image.png", help="输入 RGB 图像路径")
    parser.add_argument("--out", default="outputs/real_input", help="输出目录")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input image not found: {input_path}. Creating an example image automatically.")
        create_example_image(input_path, size=128)
    run_pipeline(args.config, args.out, input_path)


if __name__ == "__main__":
    main()
