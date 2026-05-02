from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.experiments.final_comparison import run_final_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final method comparison from output roots.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--methods", default="configs/method_catalog.yaml")
    parser.add_argument("--outputs-config", default="configs/final_method_outputs.yaml")
    parser.add_argument("--config", default="configs/final_ours.yaml")
    parser.add_argument("--out", default="outputs/final_comparison")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    with open(args.outputs_config, "r", encoding="utf-8") as f:
        outputs = (yaml.safe_load(f) or {}).get("method_outputs", {})
    result = run_final_comparison(args.manifest, args.methods, outputs, cfg, args.out, max_images=args.max_images)
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    print(f"methods evaluated: {len(result['method_summary'])}")
    print(f"final comparison saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
