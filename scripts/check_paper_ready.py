from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.paper_readiness import check_paper_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Check paper experiment readiness.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--profile-file", default=None)
    parser.add_argument("--out", default="outputs/check_paper_ready")
    args = parser.parse_args()
    path = Path(args.profile_file) if args.profile_file else Path("configs/paper") / f"{args.profile}.yaml"
    result = check_paper_ready(path, args.out)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # readiness 报告用于正式实验前检查，warning 不会阻止 debug profile 运行。
    (out / "paper_ready.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ready: {result['ready']}")
    print(f"warnings: {len(result['warnings'])} errors: {len(result['errors'])}")


if __name__ == "__main__":
    main()
