from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cache.attach import apply_frontend_cache_config, attach_cache_to_dataset
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.manifest import load_manifest
from shapesplat.editing.suite import run_edit_suite_for_scene, summarize_edit_metrics
from shapesplat.evaluation.report import save_metrics_csv, save_metrics_json
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.utils.seed import seed_everything


def run_edit_dataset(
    config_path,
    manifest_path,
    out_dir,
    max_images=None,
    max_objects=2,
    ops=None,
    skip_existing=False,
    save_visuals=True,
    use_frontend_cache=False,
    frontend_cache_root=None,
    frontend_cache_manifest=None,
    save_frontend_cache=False,
    frontend_cache_out=None,
):
    """批量编辑稳定性评估；单图失败不会中断整个 dataset。"""

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
    # 将 cache manifest/root 绑定到每条记录，后续 build_frontend 会自动加载。
    cache_cfg = cfg.get("frontend_cache", {})
    attach_cache_to_dataset(
        records,
        cache_manifest=frontend_cache_manifest or cache_cfg.get("cache_manifest"),
        cache_root=frontend_cache_root or frontend_cache_out or cache_cfg.get("cache_root"),
    )
    if max_images is not None:
        records = records[: max(0, int(max_images))]
    out = Path(out_dir)
    rows = []
    for idx, record in enumerate(records, start=1):
        image_dir = out / "per_image" / record.image_id
        if skip_existing and (image_dir / "edit_metrics.json").exists():
            rows.extend(json.loads((image_dir / "edit_metrics.json").read_text(encoding="utf-8")))
            continue
        try:
            print(f"[{idx}/{len(records)}] edit: {record.image_id}")
            cfg_img = dict(cfg)
            cfg_img["frontend"] = dict(cfg.get("frontend", {}))
            cfg_img["frontend"]["mask_path"] = record.metadata.get("mask_path")
            image = load_image(record.image_path, size=int(cfg["image"]["size"]))
            front = build_frontend(
                image,
                cfg_img,
                record=record,
                cache_dir=record.metadata.get("frontend_cache_dir"),
                use_cache=use_frontend_cache or bool(cfg.get("frontend_cache", {}).get("use_cache", False)),
                save_cache=save_frontend_cache or bool(cfg.get("frontend_cache", {}).get("save_cache", False)),
            )
            trainer = Trainer(front, cfg_img)
            trainer.train()
            object_ids = list(range(min(len(trainer.scene.objects), int(max_objects))))
            img_rows = run_edit_suite_for_scene(trainer.scene, trainer.renderer, front, image_dir, object_ids=object_ids, edit_ops=ops, save_visuals=save_visuals, cfg=cfg_img)
            for row in img_rows:
                rows.append({"image_id": record.image_id, **row})
        except Exception as exc:
            image_dir.mkdir(parents=True, exist_ok=True)
            err = {"image_id": record.image_id, "status": "failed", "error": str(exc)}
            (image_dir / "error.json").write_text(json.dumps(err, indent=2, ensure_ascii=False), encoding="utf-8")
            rows.append(err)
            print(f"Warning: edit failed on {record.image_id}: {exc}")
    summary = summarize_edit_metrics(rows)
    save_metrics_json(rows, out / "edit_per_image.json")
    save_metrics_csv(rows, out / "edit_per_image.csv")
    save_metrics_json(summary, out / "edit_summary.json")
    save_metrics_csv(list(summary.values()), out / "edit_summary.csv")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run object editing suite on a manifest dataset.")
    parser.add_argument("--config", default="configs/editing.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/edit_dataset")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=2)
    parser.add_argument("--ops", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--use-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-root", default=None)
    parser.add_argument("--frontend-cache-manifest", default=None)
    parser.add_argument("--save-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-out", default=None)
    parser.add_argument("--no-run-metadata", action="store_true")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    args = parser.parse_args()
    ops = [x.strip() for x in args.ops.split(",") if x.strip()] if args.ops else None
    rows = run_edit_dataset(
        args.config,
        args.manifest,
        args.out,
        max_images=args.max_images,
        max_objects=args.max_objects,
        ops=ops,
        skip_existing=args.skip_existing,
        save_visuals=not args.no_visuals,
        use_frontend_cache=args.use_frontend_cache,
        frontend_cache_root=args.frontend_cache_root,
        frontend_cache_manifest=args.frontend_cache_manifest,
        save_frontend_cache=args.save_frontend_cache,
        frontend_cache_out=args.frontend_cache_out,
    )
    print(f"edit dataset rows: {len(rows)}")
    print(f"outputs saved to: {Path(args.out).resolve()}")
    if not args.no_run_metadata:
        try:
            finalize_run_outputs(args.out, args.config, "edit_dataset", manifest_path=args.manifest, registry_path=args.registry)
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
