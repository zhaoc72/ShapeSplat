from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.experiments.co3dv2_highres_readiness import check_co3dv2_highres_ready, save_co3dv2_highres_readiness_report
from shapesplat.reconstruction.ours_runner import run_ours_benchmark
from shapesplat.runtime.cli import add_runtime_args, apply_runtime_cli_overrides, prepare_runtime_for_run, runtime_overrides_from_args
from shapesplat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-resolution CO3Dv2 Ours diagnostics.")
    parser.add_argument("--config", default="configs/final_ours_co3dv2_highres.yaml")
    parser.add_argument("--manifest", default="data/co3dv2_single_benchmark/manifest.csv")
    parser.add_argument("--out", default="outputs/ours_co3dv2_vits16_highres")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--frontend-cache-manifest", default="outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv")
    parser.add_argument("--strict-ready", action="store_true")
    add_runtime_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, runtime_overrides_from_args(args))
    apply_runtime_cli_overrides(cfg, args)
    cfg.setdefault("frontend_cache", {})["use_cache"] = True
    cfg["frontend_cache"]["cache_manifest"] = args.frontend_cache_manifest
    seed_everything(int(cfg["seed"]))
    prepare_runtime_for_run(cfg, args.out, save_summary=args.runtime_summary)

    ready = check_co3dv2_highres_ready(cfg, args.manifest, args.frontend_cache_manifest, strict=args.strict_ready)
    save_co3dv2_highres_readiness_report(ready, Path(args.out) / "readiness")
    if args.strict_ready and not ready["ready"]:
        raise SystemExit(1)

    # 中文注释：该脚本是 high-res CO3Dv2 diagnostic 入口；SoftRenderer/ToyShapeBank warning 会留在 readiness 与 diagnostics 中。
    rows = run_ours_benchmark(
        args.manifest,
        cfg,
        args.out,
        max_images=args.max_images,
        use_frontend_cache=True,
        frontend_cache_manifest=args.frontend_cache_manifest,
    )
    print(f"co3dv2 highres ours rows: {len(rows)}")
    print(f"outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
