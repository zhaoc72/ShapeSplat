from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.final_paper_runner import load_final_profile, run_final_paper_experiment
from shapesplat.reporting.final_report import generate_final_report
from shapesplat.reporting.final_tables import export_final_tables
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.runtime.cli import add_runtime_args


def main() -> None:
    # 中文注释：final paper 脚本只是统一编排已有实验入口，不新增算法逻辑。
    parser = argparse.ArgumentParser(description="Run final ShapeSplat++ paper experiment pack.")
    parser.add_argument("--profile", default="configs/paper/final_debug.yaml")
    parser.add_argument("--out", default="outputs/final_paper_debug")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", dest="stop_on_failure", action="store_true", default=True)
    parser.add_argument("--no-stop-on-failure", dest="stop_on_failure", action="store_false")
    parser.add_argument("--strict-ready", action="store_true")
    parser.add_argument("--generate-report", action="store_true")
    parser.add_argument("--generate-tables", action="store_true")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    parser.add_argument("--no-run-metadata", action="store_true")
    add_runtime_args(parser)
    args = parser.parse_args()
    profile = load_final_profile(args.profile)
    if args.strict_ready:
        # 中文注释：strict-ready 用于投稿前把 stub/toy/soft fallback 提升为阻塞项。
        profile.setdefault("readiness", {})["strict"] = True
    runtime_cli: list[str] = []
    if args.device:
        runtime_cli += ["--device", args.device]
    if args.cuda_device is not None:
        runtime_cli += ["--cuda-device", str(args.cuda_device)]
    if args.require_cuda:
        runtime_cli.append("--require-cuda")
    if args.allow_cpu_fallback:
        runtime_cli.append("--allow-cpu-fallback")
    if args.mixed_precision:
        runtime_cli.append("--mixed-precision")
    if args.runtime_summary:
        runtime_cli.append("--runtime-summary")
    if runtime_cli:
        # 中文注释：final runner 通过子命令参数把 GPU runtime 要求传给 Ours 入口。
        profile["runtime_cli_args"] = runtime_cli
    summary = run_final_paper_experiment(profile, args.out, dry_run=args.dry_run, stop_on_failure=args.stop_on_failure)
    root = Path(args.out)
    if args.generate_tables and not args.dry_run:
        # 中文注释：表格导出复用 final comparison 的 method summary。
        comparison_summary = root / profile.get("comparison", {}).get("output_dir", "comparison") / "final_method_summary.json"
        if comparison_summary.exists():
            export_final_tables(comparison_summary, root / profile.get("tables", {}).get("output_dir", "tables"))
    if args.generate_report and not args.dry_run:
        # 中文注释：报告是最终实验整理稿，debug profile 不代表投稿结果。
        generate_final_report(root, root / profile.get("report", {}).get("output_dir", "report"), profile.get("report", {}).get("title", "ShapeSplat++ Final Paper Report"))
    if not args.no_run_metadata:
        try:
            finalize_run_outputs(args.out, args.profile, "final_paper", registry_path=args.registry, status=summary.get("status", "success"))
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")
    print(summary)
    print(f"final run summary: {(Path(args.out) / 'final_run_summary.json').resolve()}")


if __name__ == "__main__":
    main()
