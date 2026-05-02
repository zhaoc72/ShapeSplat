from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.co3dv2_highres_readiness import check_co3dv2_highres_ready, save_co3dv2_highres_readiness_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CO3Dv2 high-resolution diagnostic readiness.")
    parser.add_argument("--config", default="configs/final_ours_co3dv2_highres.yaml")
    parser.add_argument("--manifest", default="data/co3dv2_single_benchmark/manifest.csv")
    parser.add_argument("--cache-manifest", default=None)
    parser.add_argument("--out", default="outputs/check_co3dv2_highres_ready")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    # 中文注释：脚本只做 readiness 检查，不运行重建；strict 下会把 toy/soft/missing cache 视为错误。
    report = check_co3dv2_highres_ready(args.config, args.manifest, cache_manifest=args.cache_manifest, strict=args.strict)
    save_co3dv2_highres_readiness_report(report, args.out)
    print(json.dumps({"ready": report["ready"], "num_errors": len(report["errors"]), "num_warnings": len(report["warnings"]), "out": str(Path(args.out).resolve())}, indent=2, ensure_ascii=False))
    if args.strict and not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
