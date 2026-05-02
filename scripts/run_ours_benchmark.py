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
from shapesplat.reconstruction.ours_runner import run_ours_benchmark
from shapesplat.reconstruction.readiness import check_ours_core_ready
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.runtime.cli import add_runtime_args, apply_runtime_cli_overrides, prepare_runtime_for_run, runtime_overrides_from_args
from shapesplat.utils.seed import seed_everything


def _maybe_apply_debug_iteration_cap(cfg: dict, strict_ready: bool) -> None:
    """默认 fallback 场景自动缩短训练，避免 smoke 命令跑成正式长实验。

    配置了真实 frontend cache、prepared shape bank 或禁用 soft fallback 时不触发；
    strict-ready 也不触发，因为 strict 模式应暴露正式配置问题。
    """
    ours = cfg.get("ours", {})
    debug_cfg = cfg.setdefault("debug", {})
    debug_cfg["debug_iteration_cap_applied"] = False
    # 中文注释：debug cap 只用于 smoke test；CO3Dv2 high-res 配置可显式关闭，
    # 避免训练被压到 3/3/3/2 后误判为高分辨率质量问题。
    if debug_cfg.get("allow_debug_iteration_cap") is False:
        return
    if strict_ready or not ours.get("auto_debug_iteration_cap", False):
        return
    cache = cfg.get("frontend_cache", {})
    shape = cfg.get("shape_bank", {})
    renderer = cfg.get("renderer", {})
    fallback_debug = (
        not (cache.get("cache_manifest") or cache.get("cache_root"))
        and not shape.get("root")
        and renderer.get("fallback_to_soft", True)
    )
    if not fallback_debug:
        return
    cap = ours.get("debug_iteration_cap", {})
    for key, value in cap.items():
        if key in cfg.get("training", {}):
            cfg["training"][key] = int(value)
    cfg["training"]["log_every"] = max(1, int(cfg["training"].get("log_every", 1)))
    debug_cfg["debug_iteration_cap_applied"] = True
    print("warning: applying Ours debug iteration cap because this run uses fallback debug components.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ Ours on a benchmark v2 manifest.")
    parser.add_argument("--config", default="configs/final_ours.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/ours_benchmark")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--use-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-manifest", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--strict-ready", action="store_true")
    parser.add_argument("--no-run-metadata", action="store_true")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    add_runtime_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, runtime_overrides_from_args(args))
    # 中文注释：Ours benchmark 支持强制 CUDA，避免误把正式实验跑到 CPU。
    apply_runtime_cli_overrides(cfg, args)
    if args.frontend_cache_manifest:
        cfg.setdefault("frontend_cache", {})["cache_manifest"] = args.frontend_cache_manifest
    if args.use_frontend_cache:
        cfg.setdefault("frontend_cache", {})["use_cache"] = True
    _maybe_apply_debug_iteration_cap(cfg, args.strict_ready)
    seed_everything(int(cfg["seed"]))

    ready = check_ours_core_ready(cfg, args.manifest, strict=args.strict_ready)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    prepare_runtime_for_run(cfg, args.out, save_summary=args.runtime_summary)
    from shapesplat.utils.logging import save_json

    save_json(ready, Path(args.out) / "ours_core_ready.json")
    for warning in ready["warnings"]:
        print(f"warning: {warning}")
    if args.strict_ready and not ready["ready"]:
        for error in ready["errors"]:
            print(f"error: {error}")
        raise SystemExit(1)

    rows = run_ours_benchmark(
        args.manifest,
        cfg,
        args.out,
        max_images=args.max_images,
        split=args.split,
        subset=args.subset,
        use_frontend_cache=args.use_frontend_cache or bool(cfg.get("frontend_cache", {}).get("use_cache", False)),
        frontend_cache_manifest=args.frontend_cache_manifest or cfg.get("frontend_cache", {}).get("cache_manifest"),
        skip_existing=args.skip_existing,
        save_checkpoint=args.save_checkpoint,
    )
    import json

    summary_path = Path(args.out) / "ours_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            print_metrics(json.load(f))
    print(f"ours benchmark rows: {len(rows)}")
    print(f"ours outputs saved to: {Path(args.out).resolve()}")
    if not args.no_run_metadata:
        try:
            finalize_run_outputs(
                out_dir=args.out,
                config_path=args.config,
                run_type="ours_benchmark",
                manifest_path=args.manifest,
                registry_path=args.registry,
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
