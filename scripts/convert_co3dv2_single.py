from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.datasets.benchmark.summary_v2 import save_benchmark_summary, summarize_benchmark_v2
from shapesplat.datasets.benchmark.validator_v2 import save_benchmark_v2_validation, validate_benchmark_v2
from shapesplat.datasets.converters.co3dv2_single import CO3Dv2SingleConverter


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert local CO3Dv2 single subset to benchmark v2.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="data/co3dv2_single_benchmark")
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--split", default="test")
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prefer-set-list", default=None)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--sequence", action="append", default=None)
    args = parser.parse_args()
    cfg = {
        "max_categories": args.max_categories,
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "frame_stride": args.frame_stride,
        "split": args.split,
        "copy_files": args.copy_files,
        "overwrite": args.overwrite,
        "prefer_set_list": args.prefer_set_list,
        "categories": args.category,
        "sequences": args.sequence,
    }
    # 中文注释：小规模转换优先，避免第一次把整个 CO3Dv2 子集复制过大。
    manifest = CO3Dv2SingleConverter().convert(args.root, args.out, cfg)
    report = validate_benchmark_v2(manifest, check_optional_gt=False)
    save_benchmark_v2_validation(report, Path(args.out) / "validation")
    summary = summarize_benchmark_v2(manifest)
    save_benchmark_summary(summary, Path(args.out) / "summary")
    print(f"benchmark manifest: {manifest}")
    print(f"valid: {report['valid']} rows: {report['num_rows']} failed: {report['num_failed']}")


if __name__ == "__main__":
    main()
