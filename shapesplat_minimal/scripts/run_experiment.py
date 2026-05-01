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

from shapesplat.experiments.orchestrator import load_preset, run_experiment_plan
from shapesplat.experiments.readiness import check_experiment_ready
from shapesplat.reproducibility.finalize import finalize_run_outputs


def _preset_path(args) -> Path:
    return Path(args.preset_file) if args.preset_file else ROOT / "configs" / "presets" / f"{args.preset}.yaml"


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ShapeSplat++ experiment launcher.")
    parser.add_argument("--preset", default="minimal")
    parser.add_argument("--preset-file", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", dest="stop_on_failure", action="store_true")
    parser.add_argument("--no-stop-on-failure", dest="stop_on_failure", action="store_false")
    parser.set_defaults(stop_on_failure=True)
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    parser.add_argument("--no-run-metadata", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    preset_path = _preset_path(args)
    plan = load_preset(preset_path)
    out_dir = Path(args.out or plan.default_out)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context = {
        "out": str(out_dir),
        "preset": plan.name,
        "project_root": str(ROOT),
        "timestamp": timestamp,
        "manifest": args.manifest or "",
        "input": args.input or "",
        "mask": args.mask or "",
    }
    context.update(_parse_set(args.overrides))
    out_dir.mkdir(parents=True, exist_ok=True)

    readiness = check_experiment_ready(preset_path, out_dir, context)
    _save_json(readiness, out_dir / "readiness.json")
    if not readiness["ready"]:
        print("Experiment is not ready:")
        for err in readiness["errors"]:
            print(f"- {err}")
        raise SystemExit(2)

    summary = run_experiment_plan(plan, out_dir, context, dry_run=args.dry_run, stop_on_failure=args.stop_on_failure)
    print(json.dumps({k: summary[k] for k in ["name", "status", "num_steps", "num_success", "num_failed"]}, indent=2, ensure_ascii=False))

    if not args.no_run_metadata:
        try:
            # orchestrator 只记录统一入口本身；各 step 的 stdout/stderr 已在 logs/ 中。
            finalize_run_outputs(
                out_dir=out_dir,
                config_path=None,
                run_type=f"experiment_{plan.name}",
                input_path=args.input,
                manifest_path=args.manifest,
                registry_path=args.registry,
                status=summary.get("status", "success"),
                notes={"preset_path": str(preset_path), "dry_run": args.dry_run},
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()

