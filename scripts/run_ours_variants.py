from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.evaluation.report import print_metrics
from shapesplat.reconstruction.ours_runner import run_ours_variants_benchmark
from shapesplat.runtime.cli import add_runtime_args, apply_runtime_cli_overrides, prepare_runtime_for_run, runtime_overrides_from_args
from shapesplat.utils.seed import seed_everything


def _maybe_apply_debug_iteration_cap(cfg: dict) -> None:
    """在没有真实 cache/shape bank/renderer 的 debug 环境中缩短 variants smoke run。"""
    ours = cfg.get("ours", {})
    cache = cfg.get("frontend_cache", {})
    shape = cfg.get("shape_bank", {})
    renderer = cfg.get("renderer", {})
    if not ours.get("auto_debug_iteration_cap", False):
        return
    if cache.get("cache_manifest") or cache.get("cache_root") or shape.get("root") or not renderer.get("fallback_to_soft", True):
        return
    for key, value in (ours.get("debug_iteration_cap", {}) or {}).items():
        if key in cfg.get("training", {}):
            cfg["training"][key] = int(value)
    print("warning: applying Ours debug iteration cap for variant smoke run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ Ours variants on a benchmark.")
    parser.add_argument("--config", default="configs/final_ours.yaml")
    parser.add_argument("--variants", default="configs/ours_variants.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/ours_variants")
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--use-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-manifest", default=None)
    add_runtime_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, runtime_overrides_from_args(args))
    # 中文注释：variants 也支持 GPU runtime 参数，避免消融实验跑到意外设备。
    apply_runtime_cli_overrides(cfg, args)
    if args.use_frontend_cache:
        cfg.setdefault("frontend_cache", {})["use_cache"] = True
    if args.frontend_cache_manifest:
        cfg.setdefault("frontend_cache", {})["cache_manifest"] = args.frontend_cache_manifest
    _maybe_apply_debug_iteration_cap(cfg)
    seed_everything(int(cfg["seed"]))
    prepare_runtime_for_run(cfg, args.out, save_summary=args.runtime_summary)

    run_ours_variants_benchmark(
        args.manifest,
        cfg,
        args.variants,
        args.out,
        variant_names=args.variant,
        max_images=args.max_images,
        use_frontend_cache=args.use_frontend_cache or bool(cfg.get("frontend_cache", {}).get("use_cache", False)),
        frontend_cache_manifest=args.frontend_cache_manifest or cfg.get("frontend_cache", {}).get("cache_manifest"),
    )
    import json

    summary_path = Path(args.out) / "variant_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        print_metrics({"num_variants": len(rows)})
        for row in rows:
            print(f"{row.get('variant')}: success={row.get('num_success')} failed={row.get('num_failed')}")
    print(f"variant outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
