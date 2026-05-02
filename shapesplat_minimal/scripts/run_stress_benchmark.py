from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.baselines.load_outputs import load_baseline_output
from shapesplat.benchmarks.stress_metadata import load_stress_metadata
from shapesplat.benchmarks.stress_metrics import compute_stress_metrics
from shapesplat.benchmarks.stress_report import save_stress_summary
from shapesplat.cache.attach import apply_frontend_cache_config, attach_cache_to_dataset
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.manifest import load_manifest
from shapesplat.experiments.comparison_runner import run_comparison_for_image
from shapesplat.experiments.single_image import run_single_image_experiment
from shapesplat.frontend.file_mask_loader import load_mask_file
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.utils.seed import seed_everything


def _render_from_output(output_dir: Path, masks: torch.Tensor):
    ownership_path = output_dir / "ownership.npy"
    ownership = torch.from_numpy(np.load(ownership_path)).float() if ownership_path.exists() else masks.float()
    depth = torch.ones(masks.shape[-2:], dtype=torch.float32)
    return SimpleNamespace(ownership=ownership, depth=depth)


def _metadata_path(record) -> Path:
    path = record.metadata.get("metadata_path")
    if not path:
        raise FileNotFoundError(f"record has no metadata_path: {record.image_id}")
    return Path(path)


def run_stress_benchmark(
    config_path: str | Path,
    manifest_path: str | Path,
    out_dir: str | Path,
    max_images: int | None = None,
    run_comparison: bool = False,
    run_dummy_baselines: bool = True,
    save_visuals: bool = True,
    use_frontend_cache: bool = False,
    frontend_cache_root: str | Path | None = None,
    frontend_cache_manifest: str | Path | None = None,
    save_frontend_cache: bool = False,
    frontend_cache_out: str | Path | None = None,
) -> list[dict]:
    """运行 stress benchmark；单张失败会记录 error 并继续。"""

    cfg = load_config(config_path)
    cfg.setdefault("frontend", {})["mask_source"] = "file"
    apply_frontend_cache_config(
        cfg,
        use_cache=use_frontend_cache,
        cache_root=frontend_cache_root,
        cache_manifest=frontend_cache_manifest,
        save_cache=save_frontend_cache,
        cache_out=frontend_cache_out,
    )
    seed_everything(int(cfg.get("seed", 0)))
    records = load_manifest(manifest_path)
    # stress manifest 也可以携带 frontend_cache_dir，便于复用真实前端缓存。
    cache_cfg = cfg.get("frontend_cache", {})
    attach_cache_to_dataset(
        records,
        cache_manifest=frontend_cache_manifest or cache_cfg.get("cache_manifest"),
        cache_root=frontend_cache_root or frontend_cache_out or cache_cfg.get("cache_root"),
    )
    if max_images is not None:
        records = records[: max(0, int(max_images))]
    out = Path(out_dir)
    per_image = out / "per_image"
    rows: list[dict] = []
    for idx, record in enumerate(records, start=1):
        image_dir = per_image / record.image_id
        try:
            print(f"[{idx}/{len(records)}] stress: {record.image_id}")
            image = load_image(record.image_path, size=int(cfg["image"]["size"]))
            masks = load_mask_file(record.metadata["mask_path"], image_hw=image.shape[-2:], cfg=cfg).masks
            meta = load_stress_metadata(_metadata_path(record))
            if run_comparison:
                comp_rows = run_comparison_for_image(
                    image,
                    masks,
                    cfg,
                    image_dir,
                    record.image_id,
                    run_ours=True,
                    run_dummy_baselines=run_dummy_baselines,
                    save_visuals=save_visuals,
                    save_checkpoint=False,
                    frontend_cache_dir=record.metadata.get("frontend_cache_dir"),
                    use_frontend_cache=use_frontend_cache or bool(cfg.get("frontend_cache", {}).get("use_cache", False)),
                )
                for row in comp_rows:
                    if row.get("status") != "success":
                        rows.append({**row, "Subset": meta.subset})
                        continue
                    method_dir = Path(row["output_dir"])
                    try:
                        pred = load_baseline_output(method_dir, row["method"], record.image_id)
                        render = SimpleNamespace(ownership=pred["ownership"], depth=torch.ones_like(masks[0]))
                    except Exception:
                        render = _render_from_output(method_dir, masks)
                    stress = compute_stress_metrics(render, masks, meta)
                    rows.append({**row, **stress, "subset": meta.subset})
            else:
                ours_dir = image_dir / "ours"
                cfg_img = dict(cfg)
                cfg_img["frontend"] = dict(cfg.get("frontend", {}))
                cfg_img["frontend"]["mask_path"] = record.metadata["mask_path"]
                row = run_single_image_experiment(
                    image,
                    cfg_img,
                    ours_dir,
                    image_id=record.image_id,
                    record=record,
                    save_visuals=save_visuals,
                    save_checkpoint=False,
                    eval_metrics=True,
                    frontend_cache_dir=record.metadata.get("frontend_cache_dir"),
                    use_frontend_cache=use_frontend_cache or bool(cfg.get("frontend_cache", {}).get("use_cache", False)),
                    save_frontend_cache=save_frontend_cache or bool(cfg.get("frontend_cache", {}).get("save_cache", False)),
                )
                stress = compute_stress_metrics(_render_from_output(ours_dir, masks), masks, meta)
                rows.append({**row, **stress, "method": "ours", "subset": meta.subset})
        except Exception as exc:
            image_dir.mkdir(parents=True, exist_ok=True)
            err = {"image_id": record.image_id, "status": "failed", "error": str(exc), "Subset": record.metadata.get("subset", "unknown")}
            (image_dir / "error.json").write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(err)
            print(f"Warning: stress sample failed: {record.image_id}: {exc}")
    save_stress_summary(rows, out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ synthetic stress benchmark.")
    parser.add_argument("--config", default="configs/stress_benchmark.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/stress_benchmark")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--no-dummy-baselines", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--use-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-root", default=None)
    parser.add_argument("--frontend-cache-manifest", default=None)
    parser.add_argument("--save-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-out", default=None)
    parser.add_argument("--no-run-metadata", action="store_true")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    args = parser.parse_args()
    run_stress_benchmark(
        args.config,
        args.manifest,
        args.out,
        max_images=args.max_images,
        run_comparison=args.run_comparison,
        run_dummy_baselines=not args.no_dummy_baselines,
        save_visuals=args.save_visuals,
        use_frontend_cache=args.use_frontend_cache,
        frontend_cache_root=args.frontend_cache_root,
        frontend_cache_manifest=args.frontend_cache_manifest,
        save_frontend_cache=args.save_frontend_cache,
        frontend_cache_out=args.frontend_cache_out,
    )
    if not args.no_run_metadata:
        try:
            finalize_run_outputs(args.out, args.config, "stress_benchmark", manifest_path=args.manifest, registry_path=args.registry)
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
