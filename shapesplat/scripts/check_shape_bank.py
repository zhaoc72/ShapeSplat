from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.shape_prior.retrieval import retrieve_shapes
from shapesplat.shape_prior.shape_bank_backend import build_shape_bank
from shapesplat.utils.seed import seed_everything


def _tensor_stats_list(x: torch.Tensor) -> list:
    return x.detach().cpu().tolist()


def _infer_file_descriptor_dim(root: str | None) -> int | None:
    """从 file shape bank 中轻量推断 descriptor 维度，方便 stub DINO 对齐维度。

    用户常会用 minimal.yaml 临时覆盖 --backend file；这时 config 里没有写
    descriptor_dim。为了让检查脚本更顺手，我们读取第一个带 descriptor 的 npz。
    """
    if root is None:
        return None
    root_path = Path(root)
    if not root_path.exists():
        return None
    for path in sorted(root_path.glob("*.npz")):
        data = __import__("numpy").load(path, allow_pickle=True)
        if "descriptor" in data:
            return int(data["descriptor"].reshape(-1).shape[0])
        if "descriptors" in data:
            return int(data["descriptors"].shape[-1])
    return None


def check_shape_bank(config: str, backend: str | None, root: str | None, out: str) -> None:
    """检查 shape bank 是否能加载并完成 descriptor retrieval。

    这个脚本只验证 shape bank / retrieval 路径，不检查 3D 重建质量。
    """
    cfg = load_config(config)
    seed_everything(cfg["seed"])
    if backend is not None:
        cfg.setdefault("shape_bank", {})["backend"] = backend
    if root is not None:
        cfg.setdefault("shape_bank", {})["root"] = root
    if cfg.get("shape_bank", {}).get("backend") == "file" and cfg["shape_bank"].get("descriptor_dim") is None:
        inferred_dim = _infer_file_descriptor_dim(cfg["shape_bank"].get("root"))
        if inferred_dim is not None:
            cfg["shape_bank"]["descriptor_dim"] = inferred_dim
            cfg.setdefault("frontend", {})["dino_feature_dim"] = inferred_dim
            print(f"inferred file shape descriptor dim: {inferred_dim}")

    device = torch.device(cfg["device"])
    image = make_synthetic_image(int(cfg["image"]["size"])).to(device)
    front = build_frontend(image, cfg)
    bank = build_shape_bank(cfg, descriptor_dim=front.descriptors.shape[1], device=device)
    retrieved, weights, confidence = retrieve_shapes(
        front.descriptors,
        bank,
        top_k=cfg["retrieval"]["top_k"],
        use_multi_view_descriptors=cfg["retrieval"].get("use_multi_view_descriptors", True),
        temperature=cfg["retrieval"].get("temperature", 0.07),
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_rows = []
    for asset in bank.assets:
        row = {
            "shape_id": asset.shape_id,
            "points_shape": list(asset.points.shape),
            "descriptor_shape": list(asset.descriptor.shape) if asset.descriptor is not None else None,
            "descriptors_shape": list(asset.descriptors.shape) if asset.descriptors is not None else None,
            "category": asset.category,
            "metadata": asset.metadata,
        }
        asset_rows.append(row)
        print(f"asset {asset.shape_id}: points={row['points_shape']} descriptor={row['descriptor_shape']}")

    retrieval_rows = []
    for i, row in enumerate(retrieved):
        ids = [asset.shape_id for asset in row]
        retrieval_rows.append(
            {
                "object_id": i,
                "shape_ids": ids,
                "weights": _tensor_stats_list(weights[i]),
                "confidence": float(confidence[i].detach().cpu()),
            }
        )
        print(f"object {i}: top-k={ids} confidence={retrieval_rows[-1]['confidence']:.4f}")

    stats = {
        "backend": cfg["shape_bank"]["backend"],
        "num_assets": len(bank.assets),
        "descriptors_shape": list(front.descriptors.shape),
        "assets": asset_rows,
    }
    results = {
        "weights": _tensor_stats_list(weights),
        "confidence": _tensor_stats_list(confidence),
        "retrieval": retrieval_rows,
    }
    (out_dir / "shape_bank_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "retrieval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"backend: {cfg['shape_bank']['backend']}")
    print(f"num assets: {len(bank.assets)}")
    print(f"descriptors shape: {tuple(front.descriptors.shape)}")
    print(f"shape bank check ok: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ shape bank backend and retrieval.")
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="outputs/check_shape_bank")
    args = parser.parse_args()
    check_shape_bank(args.config, args.backend, args.root, args.out)


if __name__ == "__main__":
    main()
