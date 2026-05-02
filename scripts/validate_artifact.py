from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.validation.command_matrix import run_command_matrix
from shapesplat.validation.project_health import check_project_health


def main() -> None:
    parser = argparse.ArgumentParser(description="Run artifact readiness validation.")
    parser.add_argument("--matrix", default="configs/command_matrix.yaml")
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", dest="stop_on_failure", action="store_true", default=True)
    parser.add_argument("--no-stop-on-failure", dest="stop_on_failure", action="store_false")
    parser.add_argument("--out", default="outputs/artifact_validation")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # artifact validation 先做静态健康检查，再按命令矩阵跑 smoke commands。
    health = check_project_health()
    rows = run_command_matrix(
        args.matrix,
        groups=args.groups,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure,
        context={"validation_out": str(out)},
    )
    summary = {
        "healthy": health["healthy"],
        "num_commands": len(rows),
        "num_success": sum(1 for r in rows if r["status"] == "success"),
        "num_failed": sum(1 for r in rows if r["status"] == "failed"),
        "num_dry_run": sum(1 for r in rows if r["status"] == "dry_run"),
        "valid": health["healthy"] and all(r["status"] in {"success", "dry_run"} for r in rows),
    }
    (out / "project_health.json").write_text(json.dumps(health, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "command_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "artifact_validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
