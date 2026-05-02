from __future__ import annotations

import traceback
from pathlib import Path

from shapesplat.evaluation.report import load_metrics_json, save_metrics_json
from shapesplat.experiments.single_image import run_single_image_experiment


def run_batch_experiment(
    dataset,
    cfg: dict,
    out_dir: str | Path,
    max_images: int | None = None,
    skip_existing: bool = False,
    save_visuals: bool = True,
    save_checkpoint: bool = False,
    eval_metrics: bool = True,
    use_frontend_cache: bool = False,
    save_frontend_cache: bool = False,
) -> list[dict]:
    """逐图运行 batch experiment。

    单张图失败时不会中断整个 batch；失败信息会写入该 image_id 目录下的
    error.json，并在 summary rows 中标记 status=failed。
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    total = len(dataset) if max_images is None else min(len(dataset), int(max_images))
    rows: list[dict] = []

    for idx in range(total):
        item = dataset[idx]
        image_id = str(item["image_id"])
        image_out = out_root / image_id
        metrics_path = image_out / "metrics.json"
        print(f"[{idx + 1}/{total}] Running image: {image_id}")

        if skip_existing and metrics_path.exists():
            row = load_metrics_json(metrics_path)
            row.setdefault("image_id", image_id)
            row.setdefault("status", "success")
            rows.append(row)
            print(f"  skip existing: {metrics_path}")
            continue

        try:
            row = run_single_image_experiment(
                item["image"],
                cfg,
                image_out,
                image_id=image_id,
                record=item.get("record"),
                save_visuals=save_visuals,
                save_checkpoint=save_checkpoint,
                eval_metrics=eval_metrics,
                frontend_cache_dir=getattr(item.get("record"), "metadata", {}).get("frontend_cache_dir"),
                use_frontend_cache=use_frontend_cache or bool(cfg.get("frontend_cache", {}).get("use_cache", False)),
                save_frontend_cache=save_frontend_cache or bool(cfg.get("frontend_cache", {}).get("save_cache", False)),
            )
            row["image_path"] = item["image_path"]
            rows.append(row)
        except Exception as exc:
            image_out.mkdir(parents=True, exist_ok=True)
            error = {
                "image_id": image_id,
                "image_path": item.get("image_path"),
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "output_dir": str(image_out),
            }
            save_metrics_json(error, image_out / "error.json")
            rows.append(error)
            print(f"  failed: {exc}")
    return rows
