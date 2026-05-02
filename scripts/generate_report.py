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

from shapesplat.reporting.io import load_json
from shapesplat.reporting.report import generate_experiment_report
from shapesplat.reproducibility.finalize import finalize_run_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ShapeSplat++ experiment report.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="ShapeSplat++ Experiment Report")
    parser.add_argument("--no-run-metadata", action="store_true", help="不写入 run_info / registry 元数据")
    parser.add_argument("--registry", default="runs/run_registry.jsonl", help="全局 run registry 路径")
    args = parser.parse_args()

    manifest = generate_experiment_report(args.root, args.out, args.title)
    sanity = load_json(manifest["metric_sanity"])
    print(f"report: {manifest['report']}")
    print(f"metric sanity bad rows: {sanity.get('num_bad_rows', 0)}")
    print(f"failure cases: {manifest.get('num_failure_cases', 0)}")
    if not args.no_run_metadata:
        try:
            # report 本身也是一次实验产物整理，需要记录来源 root 和输出报告目录。
            finalize_run_outputs(
                out_dir=args.out,
                config_path=None,
                run_type="report",
                registry_path=args.registry,
                notes={"root": str(args.root), "title": str(args.title)},
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
