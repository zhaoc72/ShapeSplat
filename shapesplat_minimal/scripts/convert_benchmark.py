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
from shapesplat.datasets.converters.co3d_template import CO3DConverterTemplate
from shapesplat.datasets.converters.co3dv2_single import CO3Dv2SingleConverter
from shapesplat.datasets.converters.generic_folder import GenericFolderConverter
from shapesplat.datasets.converters.gso_template import GSOConverterTemplate
from shapesplat.datasets.converters.objaverse_template import ObjaverseConverterTemplate
from shapesplat.datasets.converters.pix3d_template import Pix3DConverterTemplate


CONVERTERS = {
    "generic_folder": GenericFolderConverter,
    "gso_template": GSOConverterTemplate,
    "objaverse_template": ObjaverseConverterTemplate,
    "pix3d_template": Pix3DConverterTemplate,
    "co3d_template": CO3DConverterTemplate,
    "co3dv2_single": CO3Dv2SingleConverter,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert local data to benchmark manifest v2.")
    parser.add_argument("--converter", required=True, choices=sorted(CONVERTERS))
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-dataset", default="custom")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--copy-files", action="store_true")
    args = parser.parse_args()
    converter = CONVERTERS[args.converter]()
    cfg = {
        "source_dataset": args.source_dataset,
        "overwrite": args.overwrite,
        "copy_files": args.copy_files,
        "max_categories": args.max_categories,
        "max_sequences": args.max_sequences,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "frame_stride": args.frame_stride,
    }
    if args.converter == "co3dv2_single" and args.max_categories is None and args.max_sequences is None and args.max_frames_per_sequence is None:
        # 中文注释：通用入口默认做小规模 CO3D smoke conversion，避免误扫完整真实数据集；全量请使用专用脚本显式配置。
        cfg.update({"max_categories": 3, "max_sequences": 2, "max_frames_per_sequence": 5})
    manifest = converter.convert(args.src, args.out, cfg)
    report = validate_benchmark_v2(manifest)
    save_benchmark_v2_validation(report, Path(args.out) / "validation")
    summary = summarize_benchmark_v2(manifest)
    save_benchmark_summary(summary, Path(args.out) / "summary")
    print(f"benchmark manifest: {manifest}")
    print(f"valid: {report['valid']} num_rows: {report['num_rows']}")


if __name__ == "__main__":
    main()
