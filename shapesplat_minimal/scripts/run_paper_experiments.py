from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.paper_runner import generate_paper_report, generate_paper_tables, load_paper_profile, run_paper_profile
from shapesplat.reproducibility.finalize import finalize_run_outputs


def main() -> None:
    # 论文实验入口只负责 orchestration：按 profile 调已有脚本，不改算法逻辑。
    parser = argparse.ArgumentParser(description="Run paper-style experiment profiles.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--profile-file", default=None)
    parser.add_argument("--out", default="outputs/paper_debug")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", dest="stop_on_failure", action="store_true", default=True)
    parser.add_argument("--no-stop-on-failure", dest="stop_on_failure", action="store_false")
    parser.add_argument("--generate-tables", action="store_true")
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    parser.add_argument("--no-run-metadata", action="store_true")
    args = parser.parse_args()
    profile_path = Path(args.profile_file) if args.profile_file else Path("configs/paper") / f"{args.profile}.yaml"
    profile = load_paper_profile(profile_path)
    summary = run_paper_profile(profile, args.out, dry_run=args.dry_run, stop_on_failure=args.stop_on_failure)
    if args.generate_tables:
        # 表格导出是论文草稿格式，最终投稿前仍需人工检查。
        generate_paper_tables(args.out, "configs/paper/table_columns.yaml")
    if args.generate_report:
        generate_paper_report(args.out)
    if not args.no_run_metadata:
        try:
            finalize_run_outputs(args.out, str(profile_path), "paper", registry_path=args.registry)
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")
    print(summary)
    print(f"paper outputs saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
