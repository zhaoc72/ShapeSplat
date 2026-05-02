from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.orchestrator import load_preset
from shapesplat.experiments.readiness import check_experiment_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a ShapeSplat++ experiment preset is ready.")
    parser.add_argument("--preset", default="minimal")
    parser.add_argument("--preset-file", default=None)
    parser.add_argument("--out", default="outputs/check_ready")
    parser.add_argument("--input", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    preset_path = Path(args.preset_file) if args.preset_file else ROOT / "configs" / "presets" / f"{args.preset}.yaml"
    plan = load_preset(preset_path)
    context = {
        "out": args.out,
        "preset": plan.name,
        "project_root": str(ROOT),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input": args.input or "",
        "mask": args.mask or "",
        "manifest": args.manifest or "",
    }
    result = check_experiment_ready(preset_path, args.out, context)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "readiness.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"ready: {result['ready']}")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    for error in result["errors"]:
        print(f"error: {error}")
    print(f"readiness saved to: {(out / 'readiness.json').resolve()}")


if __name__ == "__main__":
    main()

