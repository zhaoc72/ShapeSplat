from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.evaluation.report import save_metrics_csv
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.shape_prior.retrieval import retrieve_shapes
from shapesplat.shape_prior.shape_bank_backend import build_shape_bank
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid


def check_shape_retrieval(config: str, input_path: str, shape_root: str | None, out: str, top_k: int) -> list[dict]:
    """检查 image instance descriptor 到 shape bank descriptor 的 top-k 检索。"""
    cfg = load_config(config)
    if shape_root:
        cfg.setdefault("shape_bank", {})["backend"] = "file"
        cfg["shape_bank"]["root"] = shape_root
    image = load_image(input_path, cfg["image"]["size"])
    front = build_frontend(image, cfg)
    bank = build_shape_bank(cfg, descriptor_dim=front.descriptors.shape[1], device=front.descriptors.device)
    retrieved, weights, confidence = retrieve_shapes(
        front.descriptors,
        bank,
        top_k=top_k,
        use_multi_view_descriptors=cfg["retrieval"].get("use_multi_view_descriptors", True),
        temperature=cfg["retrieval"].get("temperature", 0.07),
    )
    rows = []
    for i, assets in enumerate(retrieved):
        rows.append(
            {
                "object_id": i,
                "shape_ids": ",".join(a.shape_id for a in assets),
                "weights": ",".join(f"{float(v):.6f}" for v in weights[i].detach().cpu()),
                "confidence": float(confidence[i].detach().cpu()),
            }
        )
        print(f"object {i}: {[a.shape_id for a in assets]} confidence={rows[-1]['confidence']:.4f}")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_mask_grid(front.masks, out_dir / "masks.png")
    save_input_with_mask_overlay(front.image, front.masks, out_dir / "input_mask_overlay.png")
    payload = {
        "num_image_objects": int(front.descriptors.shape[0]),
        "descriptor_dim": int(front.descriptors.shape[1]),
        "num_shapes": len(bank.assets),
        "rows": rows,
        "weights": weights.detach().cpu().tolist(),
        "confidence": confidence.detach().cpu().tolist(),
    }
    (out_dir / "retrieval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_metrics_csv(rows, out_dir / "retrieval_table.csv")
    print(f"shape retrieval check ok: {out_dir}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check image-to-shape retrieval.")
    parser.add_argument("--config", default="configs/file_shape_bank.yaml")
    parser.add_argument("--input", default="examples/test_image.png")
    parser.add_argument("--shape-root", default=None)
    parser.add_argument("--out", default="outputs/check_shape_retrieval")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    check_shape_retrieval(args.config, args.input, args.shape_root, args.out, args.top_k)


if __name__ == "__main__":
    main()
