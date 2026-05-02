from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.shape_prior.toy_shape_bank import ToyShapeBank


def create_toy_shape_bank(out: str | Path, num_points: int = 512, descriptor_dim: int = 16) -> Path:
    """创建 file backend 可读取的 toy shape bank。

    这些 npz 只是用于 FileShapeBank 的 smoke test，不是真实实验数据。正式实验应使用
    train/test instance-disjoint 的真实 shape bank，并预计算真实多视角 DINO descriptor。
    """
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    bank = ToyShapeBank(descriptor_dim=descriptor_dim, device=torch.device("cpu"), points_per_shape=num_points)
    for asset in bank.assets:
        np.savez(
            out_path / f"{asset.shape_id}.npz",
            points=asset.points.cpu().numpy().astype("float32"),
            descriptor=asset.descriptor.cpu().numpy().astype("float32"),
            category=np.array(asset.category or "toy"),
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny npz shape bank for smoke tests.")
    parser.add_argument("--out", default="examples/shape_bank")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--descriptor-dim", type=int, default=16)
    args = parser.parse_args()
    out = create_toy_shape_bank(args.out, args.num_points, args.descriptor_dim)
    print(f"toy file shape bank saved to: {out}")


if __name__ == "__main__":
    main()
