from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.co3dv2_diagnostics import run_co3dv2_diagnostics


def main() -> None:
    # scripts 只保留命令行入口；可复用 diagnostics 逻辑位于 shapesplat.experiments。
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ diagnostics on converted CO3Dv2 single benchmark.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/final_ours.yaml")
    parser.add_argument("--out", default="outputs/co3dv2_diagnostics")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--use-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-manifest", default=None)
    parser.add_argument("--run-editing", action="store_true")
    parser.add_argument("--generate-report", action="store_true")
    args = parser.parse_args()
    result = run_co3dv2_diagnostics(
        args.manifest,
        args.config,
        args.out,
        max_images=args.max_images,
        use_frontend_cache=args.use_frontend_cache,
        frontend_cache_manifest=args.frontend_cache_manifest,
        run_editing=args.run_editing,
        generate_report=args.generate_report,
    )
    print(result)


if __name__ == "__main__":
    main()
