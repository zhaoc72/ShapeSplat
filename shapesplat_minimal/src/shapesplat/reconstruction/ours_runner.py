from __future__ import annotations

import json
from pathlib import Path

import torch

from shapesplat.cache.frontend_cache import load_frontend_cache_manifest
from shapesplat.data.image_io import load_image
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.evaluation.edit_metrics import compute_edit_metrics
from shapesplat.evaluation.metrics import compute_basic_metrics
from shapesplat.evaluation.report import flatten_metrics, merge_metrics, save_metrics_csv, save_metrics_json
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.reconstruction.diagnostics import compute_reconstruction_diagnostics
from shapesplat.reconstruction.output_protocol import save_ours_output
from shapesplat.reconstruction.variants import apply_variant_overrides, get_variant_by_name, load_ours_variants
from shapesplat.utils.logging import save_json


def _summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "success"]
    summary = {"num_images": len(rows), "num_success": len(ok), "num_failed": len(rows) - len(ok)}
    keys: list[str] = []
    for row in ok:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"num_masks", "num_objects"}:
                keys.append(key)
    for key in sorted(set(keys)):
        vals = [float(r[key]) for r in ok if isinstance(r.get(key), (int, float))]
        if vals:
            summary[f"{key}_mean"] = sum(vals) / len(vals)
    return summary


def _resolve_cache_dir(record, image_id: str, cache_map: dict | None, cfg: dict, explicit=None):
    if explicit:
        return explicit
    if record is not None and getattr(record, "metadata", {}).get("frontend_cache_dir"):
        return record.metadata["frontend_cache_dir"]
    if cache_map and image_id in cache_map:
        return cache_map[image_id].cache_dir
    root = cfg.get("frontend_cache", {}).get("cache_root")
    if root:
        return Path(root) / image_id
    return None


def run_ours_single(
    image: torch.Tensor,
    cfg: dict,
    out_dir: str | Path,
    image_id: str = "image",
    record=None,
    frontend_cache_dir=None,
    use_frontend_cache: bool = False,
    save_frontend_cache: bool = False,
    save_checkpoint: bool = False,
    save_visuals: bool = True,
    eval_metrics: bool = True,
) -> dict:
    """运行单张图的 Ours 主方法。

    该函数复用现有 front-end、Trainer 和 renderer，只增加正式 benchmark 需要的
    输出协议与 diagnostics。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    front = build_frontend(
        image,
        cfg,
        record=record,
        cache_dir=frontend_cache_dir,
        use_cache=use_frontend_cache,
        save_cache=save_frontend_cache,
    )
    if front.masks.shape[0] == 0:
        raise RuntimeError(f"{image_id}: front-end produced no masks.")

    trainer = Trainer(front, cfg)
    loss_log = trainer.train()
    render = trainer.render()
    save_json(loss_log, out_path / "loss_log.json")

    metrics: dict = {"image_id": image_id, "method": "ours", "variant": cfg.get("ours", {}).get("variant", cfg.get("ablation_name", "full"))}
    if eval_metrics:
        metrics = merge_metrics(
            metrics,
            compute_basic_metrics(render, front.masks),
            compute_edit_metrics(trainer.scene, trainer.renderer, front, render, cfg, object_id=0),
        )
    diagnostics = compute_reconstruction_diagnostics(trainer.scene, render, front, cfg)
    save_ours_output(
        out_path,
        front.image,
        front.masks,
        render,
        metrics,
        scene=trainer.scene,
        cfg=cfg,
        diagnostics=diagnostics if cfg.get("ours", {}).get("save_diagnostics", True) else None,
        save_checkpoint=save_checkpoint or bool(cfg.get("ours", {}).get("save_checkpoint", False)),
        image_id=image_id,
    )
    row = {
        "image_id": image_id,
        "method": "ours",
        "variant": cfg.get("ours", {}).get("variant", "full"),
        "status": "success",
        "num_masks": int(front.masks.shape[0]),
        "num_objects": int(len(trainer.scene.objects)),
        "output_dir": str(out_path),
    }
    row.update(metrics)
    row["gaussian_count_total"] = diagnostics["object_counts"]["gaussian_count_total"]
    row["renderer_backend"] = diagnostics["renderer"]["renderer_backend"]
    return row


def run_ours_benchmark(
    manifest_path: str | Path,
    cfg: dict,
    out_dir: str | Path,
    max_images: int | None = None,
    split: str | None = None,
    subset: str | None = None,
    use_frontend_cache: bool | None = None,
    frontend_cache_manifest: str | Path | None = None,
    skip_existing: bool = False,
    save_checkpoint: bool = False,
) -> list[dict]:
    """在 benchmark v2 或旧 manifest 上批量运行 Ours 主方法。"""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset_from_manifest(manifest_path, image_size=int(cfg["image"]["size"]))
    records = dataset.records
    if split:
        records = [r for r in records if r.split == split]
    if subset:
        records = [r for r in records if r.metadata.get("subset") == subset]
    if max_images is not None:
        records = records[: int(max_images)]
    dataset.records = records

    cache_map = load_frontend_cache_manifest(frontend_cache_manifest) if frontend_cache_manifest else None
    if use_frontend_cache is None:
        use_frontend_cache = bool(cfg.get("frontend_cache", {}).get("use_cache", False))

    rows: list[dict] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        image_id = item["image_id"]
        image_dir = out_root / "per_image" / image_id
        if skip_existing and (image_dir / "metrics.json").exists():
            with open(image_dir / "metrics.json", "r", encoding="utf-8") as f:
                rows.append(json.load(f))
            continue
        try:
            print(f"[{idx + 1}/{len(dataset)}] ours: {image_id}")
            cache_dir = _resolve_cache_dir(item.get("record"), image_id, cache_map, cfg)
            row = run_ours_single(
                item["image"],
                cfg,
                image_dir,
                image_id=image_id,
                record=item.get("record"),
                frontend_cache_dir=cache_dir,
                use_frontend_cache=bool(use_frontend_cache),
                save_frontend_cache=bool(cfg.get("frontend_cache", {}).get("save_cache", False)),
                save_checkpoint=save_checkpoint,
                save_visuals=bool(cfg.get("ours", {}).get("save_visuals", True)),
                eval_metrics=True,
            )
            rows.append(row)
        except Exception as exc:
            image_dir.mkdir(parents=True, exist_ok=True)
            err = {"image_id": image_id, "method": "ours", "status": "failed", "output_dir": str(image_dir), "error": str(exc)}
            save_metrics_json(err, image_dir / "error.json")
            rows.append(err)
            print(f"Warning: Ours failed on {image_id}: {exc}")

    flat = [flatten_metrics(r) for r in rows]
    save_metrics_json(rows, out_root / "ours_per_image.json")
    save_metrics_csv(flat, out_root / "ours_per_image.csv")
    summary = _summarize(flat)
    save_metrics_json(summary, out_root / "ours_summary.json")
    save_metrics_csv([summary], out_root / "ours_summary.csv")
    return rows


def run_ours_variants_benchmark(
    manifest_path,
    base_cfg,
    variants_path,
    out_dir,
    variant_names: list[str] | None = None,
    max_images: int | None = None,
    use_frontend_cache: bool | None = None,
    frontend_cache_manifest: str | Path | None = None,
) -> list[dict]:
    """批量运行 Ours 内部变体，用于正式 ablation 表。"""
    variants = load_ours_variants(variants_path)
    if variant_names:
        variants = [get_variant_by_name(variants, name) for name in variant_names]
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    variant_summary: list[dict] = []
    for variant in variants:
        name = variant.get("name", "variant")
        cfg = apply_variant_overrides(base_cfg, variant)
        rows = run_ours_benchmark(
            manifest_path,
            cfg,
            root / name,
            max_images=max_images,
            use_frontend_cache=use_frontend_cache,
            frontend_cache_manifest=frontend_cache_manifest,
        )
        for row in rows:
            row["variant"] = name
        all_rows.extend(rows)
        summary = _summarize([flatten_metrics(r) for r in rows])
        summary["variant"] = name
        variant_summary.append(summary)
    save_metrics_json(all_rows, root / "variant_per_image.json")
    save_metrics_csv([flatten_metrics(r) for r in all_rows], root / "variant_per_image.csv")
    save_metrics_json(variant_summary, root / "variant_summary.json")
    save_metrics_csv(variant_summary, root / "variant_summary.csv")
    return all_rows
