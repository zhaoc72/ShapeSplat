from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.final_paper_runner import load_final_profile, run_final_paper_experiment
from shapesplat.experiments.paper_runner import generate_paper_report, generate_paper_tables, load_paper_profile, run_paper_profile
from shapesplat.reporting.final_report import generate_final_report
from shapesplat.reporting.final_tables import export_final_tables
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.runtime.cli import add_runtime_args


def main() -> None:
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
    add_runtime_args(parser)
    args = parser.parse_args()

    profile_path = Path(args.profile_file) if args.profile_file else Path("configs/paper") / f"{args.profile}.yaml"
    profile = load_final_profile(profile_path) if profile_path.name.startswith("final_") else load_paper_profile(profile_path)

    if profile.get("final_paper") or profile.get("profile") in {"final_debug", "final_all", "final_main"}:
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
            # 中文注释：paper 入口遇到 final profile 时，把 GPU runtime 参数透传到 Ours 子命令。
            profile["runtime_cli_args"] = runtime_cli
        summary = run_final_paper_experiment(profile, args.out, dry_run=args.dry_run, stop_on_failure=args.stop_on_failure)
        if args.generate_tables and not args.dry_run:
            comparison_summary = Path(args.out) / profile.get("comparison", {}).get("output_dir", "comparison") / "final_method_summary.json"
            if comparison_summary.exists():
                export_final_tables(comparison_summary, Path(args.out) / profile.get("tables", {}).get("output_dir", "tables"))
        if args.generate_report and not args.dry_run:
            generate_final_report(
                args.out,
                Path(args.out) / profile.get("report", {}).get("output_dir", "report"),
                profile.get("report", {}).get("title", "ShapeSplat++ Final Paper Report"),
            )
    else:
        summary = run_paper_profile(profile, args.out, dry_run=args.dry_run, stop_on_failure=args.stop_on_failure)
        if args.generate_tables:
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
