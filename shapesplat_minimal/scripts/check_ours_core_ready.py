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
from shapesplat.reconstruction.readiness import check_ours_core_ready
from shapesplat.utils.logging import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether Ours core runner is ready for final benchmark.")
    parser.add_argument("--config", default="configs/final_ours.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", default="outputs/check_ours_core_ready")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = check_ours_core_ready(cfg, args.manifest, strict=args.strict)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(report, out_dir / "ours_core_ready.json")

    print(f"ready: {report['ready']}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")
    for error in report["errors"]:
        print(f"error: {error}")
    print(f"report saved to: {(out_dir / 'ours_core_ready.json').resolve()}")
    if args.strict and not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
