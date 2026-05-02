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

from shapesplat.reproducibility.finalize import finalize_run_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize an existing ShapeSplat++ run with reproducibility metadata.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-type", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--status", default="success")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    args = parser.parse_args()
    result = finalize_run_outputs(
        args.out,
        args.config,
        args.run_type,
        input_path=args.input,
        manifest_path=args.manifest,
        status=args.status,
        registry_path=args.registry,
    )
    print(f"run_id: {result['run_id']}")
    print(f"registry: {result['registry_path']}")


if __name__ == "__main__":
    main()

