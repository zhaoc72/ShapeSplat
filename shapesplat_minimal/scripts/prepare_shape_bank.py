from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.shape_prior.descriptor_precompute import precompute_shape_descriptors
from shapesplat.shape_prior.toy_shape_bank import ToyShapeBank
from shapesplat.evaluation.report import save_metrics_csv

import numpy as np
import torch


def _write_toy_bank(tmp: Path, num_points: int, descriptor_dim: int) -> None:
    """生成 toy 点云作为 shape bank 准备流程的 smoke test 输入。"""
    tmp.mkdir(parents=True, exist_ok=True)
    bank = ToyShapeBank(descriptor_dim=descriptor_dim, device=torch.device("cpu"), points_per_shape=num_points)
    for asset in bank.assets:
        np.savez(tmp / f"{asset.shape_id}.npz", points=asset.points.cpu().numpy(), category=np.array(asset.category or "toy"))


def prepare_shape_bank(source: str, input_root: str | None, out: str, num_points: int, descriptor_dim: int, descriptor_mode: str) -> list[dict]:
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / "_source_points"
    if tmp.exists():
        shutil.rmtree(tmp)
    if source == "toy":
        _write_toy_bank(tmp, num_points, descriptor_dim)
        src = tmp
    else:
        if not input_root:
            raise ValueError("--input is required when --source file")
        src = Path(input_root)
    rows = precompute_shape_descriptors(src, out_root, descriptor_dim, descriptor_mode)
    (out_root / "precompute_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_metrics_csv(rows, out_root / "precompute_summary.csv")
    if tmp.exists():
        shutil.rmtree(tmp)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a descriptor-ready npz shape bank.")
    parser.add_argument("--source", choices=["toy", "file"], default="toy")
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default="examples/shape_bank_prepared")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--descriptor-dim", type=int, default=16)
    parser.add_argument("--descriptor-mode", choices=["point_stats", "random"], default="point_stats")
    args = parser.parse_args()
    rows = prepare_shape_bank(args.source, args.input, args.out, args.num_points, args.descriptor_dim, args.descriptor_mode)
    print(f"prepared shape bank: {len(rows)} shapes -> {args.out}")


if __name__ == "__main__":
    main()
