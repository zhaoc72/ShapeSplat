from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.benchmarks.standard.builder import build_from_folder, build_same_mask_benchmark
from shapesplat.benchmarks.standard.validator import save_validation_report, validate_benchmark_manifest
from shapesplat.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standard same-mask benchmark directory.")
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default="configs/same_mask.yaml")
    args = parser.parse_args()
    if args.source_manifest:
        manifest = build_same_mask_benchmark(args.source_manifest, args.out, copy_files=args.copy_files, overwrite=args.overwrite)
    else:
        if not args.image_dir or not args.mask_dir:
            raise ValueError("provide --source-manifest or both --image-dir and --mask-dir")
        manifest = build_from_folder(args.image_dir, args.mask_dir, args.out)
    report = validate_benchmark_manifest(manifest, load_config(args.config))
    save_validation_report(report, Path(args.out) / "validation")
    print(f"benchmark manifest: {manifest}")
    print(f"valid: {report['valid']}")


if __name__ == "__main__":
    main()
