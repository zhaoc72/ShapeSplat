from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.final_paper_runner import load_final_profile
from shapesplat.experiments.final_readiness import check_final_paper_ready, save_final_readiness_report


def main() -> None:
    # 中文注释：readiness 只做正式实验前的 sanity check，不会运行重模型推理。
    parser = argparse.ArgumentParser(description="Check final paper experiment readiness.")
    parser.add_argument("--profile", default="configs/paper/final_debug.yaml")
    parser.add_argument("--out", default="outputs/check_final_ready")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    profile = load_final_profile(args.profile)
    report = check_final_paper_ready(profile, args.out, strict=args.strict)
    # 中文注释：同时保存 JSON 和 Markdown，方便机器读取和人工检查。
    save_final_readiness_report(report, args.out)
    print(f"ready: {report['ready']}")
    for w in report["warnings"]:
        print(f"warning: {w}")
    for e in report["errors"]:
        print(f"error: {e}")
    if args.strict and not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
