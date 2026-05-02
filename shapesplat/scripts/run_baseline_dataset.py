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

from shapesplat.baselines.export_inputs import export_baseline_inputs
from shapesplat.baselines.dummy_baselines import DUMMY_BASELINES, run_dummy_baseline, save_baseline_prediction
from shapesplat.baselines.evaluate_baseline import evaluate_baseline_prediction
from shapesplat.baselines.compare import compare_methods_for_image
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.manifest import load_manifest
from shapesplat.evaluation.report import flatten_metrics, save_metrics_csv, save_metrics_json
from shapesplat.frontend.file_mask_loader import load_mask_file


def run_dummy_baselines_for_image(image, masks, out_dir: str | Path, image_id: str) -> list[dict]:
    """dataset runner 内部使用的 dummy baseline 执行函数。

    这里避免 shell 调子进程，也避免真实 baseline 依赖；只用于协议 smoke test。
    """

    out_dir = Path(out_dir)
    predictions = {}
    rows = []
    for method in DUMMY_BASELINES.keys():
        pred = run_dummy_baseline(method, image, masks)
        metrics = evaluate_baseline_prediction(pred, masks, image=image)
        save_baseline_prediction(pred, out_dir / method, method, image_id, metrics=metrics)
        predictions[method] = pred
        rows.append({"method": method, **metrics})
    compare_methods_for_image(image, masks, predictions, out_dir)
    return rows


def summarize_baseline_rows(rows: list[dict]) -> dict:
    """对 baseline dataset rows 做轻量汇总。"""

    summary = {
        "num_total": len(rows),
        "num_success": sum(1 for r in rows if r.get("status") == "success"),
        "num_failed": sum(1 for r in rows if r.get("status") == "failed"),
    }
    numeric: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        for key, value in flatten_metrics(row).items():
            if isinstance(value, (int, float)) and key not in {"num_total", "num_success", "num_failed"}:
                numeric.setdefault(key, []).append(float(value))
    import math

    for key, vals in numeric.items():
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        summary[key] = mean
        summary[f"{key}_std"] = math.sqrt(var)
    return summary


def run_baseline_dataset(
    config: str,
    manifest: str,
    out: str | Path,
    max_images: int | None = None,
    skip_existing: bool = False,
    run_dummy: bool = False,
) -> list[dict]:
    """批量导出 baseline inputs，并可选运行 dummy baselines。

    单张图失败时写 error.json 并继续下一张，避免一个坏样本中断整批实验。
    """

    cfg = load_config(config)
    records = load_manifest(manifest)
    if max_images is not None:
        records = records[: max(0, int(max_images))]
    out = Path(out)
    rows: list[dict] = []
    for idx, record in enumerate(records, start=1):
        image_dir = out / record.image_id
        metrics_path = image_dir / "dummy" / "comparison.json"
        if skip_existing and metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for row in existing:
                rows.append({"image_id": record.image_id, "status": "success", **row})
            continue
        try:
            print(f"[{idx}/{len(records)}] baseline protocol: {record.image_id}")
            image = load_image(record.image_path, size=int(cfg["image"]["size"]))
            mask_path = record.metadata.get("mask_path")
            if not mask_path:
                raise FileNotFoundError(f"manifest record has no mask_path: {record.image_id}")
            mask_set = load_mask_file(mask_path, image_hw=image.shape[-2:], cfg=cfg)
            export_baseline_inputs(
                image,
                mask_set.masks,
                image_dir / "inputs",
                record.image_id,
                crop_padding=int(cfg.get("baseline", {}).get("crop_padding", 8)),
            )
            if run_dummy:
                dummy_rows = run_dummy_baselines_for_image(image, mask_set.masks, image_dir / "dummy", record.image_id)
                for row in dummy_rows:
                    rows.append({"image_id": record.image_id, "status": "success", **row})
            else:
                rows.append({"image_id": record.image_id, "status": "success", "output_dir": str(image_dir)})
        except Exception as exc:
            image_dir.mkdir(parents=True, exist_ok=True)
            err = {"image_id": record.image_id, "status": "failed", "error": str(exc)}
            with open(image_dir / "error.json", "w", encoding="utf-8") as f:
                json.dump(err, f, indent=2, ensure_ascii=False)
            rows.append(err)
            print(f"Warning: failed on {record.image_id}: {exc}")
    out.mkdir(parents=True, exist_ok=True)
    save_metrics_json(rows, out / "baseline_summary.json")
    save_metrics_csv([flatten_metrics(r) for r in rows], out / "baseline_summary.csv")
    save_metrics_json(summarize_baseline_rows(rows), out / "baseline_summary_stats.json")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export baseline inputs and optionally run dummy baselines for a manifest dataset.")
    parser.add_argument("--config", default="configs/baseline_protocol.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/baseline_dataset")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run-dummy", action="store_true")
    args = parser.parse_args()

    rows = run_baseline_dataset(
        args.config,
        args.manifest,
        args.out,
        max_images=args.max_images,
        skip_existing=args.skip_existing,
        run_dummy=args.run_dummy,
    )
    print(f"baseline rows: {len(rows)}")
    print(f"baseline outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
