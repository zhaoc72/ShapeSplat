from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.reconstruction.ours_runner import run_ours_variants_benchmark
from shapesplat.runtime.cli import add_runtime_args, apply_runtime_cli_overrides, prepare_runtime_for_run, runtime_overrides_from_args
from shapesplat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-resolution CO3Dv2 Ours variants.")
    parser.add_argument("--config", default="configs/final_ours_co3dv2_highres.yaml")
    parser.add_argument("--variants", default="configs/ours_variants.yaml")
    parser.add_argument("--manifest", default="data/co3dv2_single_benchmark/manifest.csv")
    parser.add_argument("--out", default="outputs/ours_variants_co3dv2_vits16_highres")
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--frontend-cache-manifest", default="outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv")
    add_runtime_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, runtime_overrides_from_args(args))
    apply_runtime_cli_overrides(cfg, args)
    cfg.setdefault("frontend_cache", {})["use_cache"] = True
    cfg["frontend_cache"]["cache_manifest"] = args.frontend_cache_manifest
    seed_everything(int(cfg["seed"]))
    prepare_runtime_for_run(cfg, args.out, save_summary=args.runtime_summary)
    rows = run_ours_variants_benchmark(
        args.manifest,
        cfg,
        args.variants,
        args.out,
        variant_names=args.variant,
        max_images=args.max_images,
        use_frontend_cache=True,
        frontend_cache_manifest=args.frontend_cache_manifest,
    )
    print(f"co3dv2 highres variant rows: {len(rows)}")
    print(f"outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
