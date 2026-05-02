from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cleanup.generated_artifacts import (
    delete_candidates_permanently,
    move_candidates_to_trash,
    save_cleanup_report,
    scan_generated_artifacts,
)
from shapesplat.cleanup.rules import load_cleanup_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely scan and clean generated test/debug artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/cleanup_generated_artifacts.yaml")
    parser.add_argument("--out", default="outputs/cleanup_report")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--permanent", action="store_true")
    parser.add_argument("--include-experiment-outputs", action="store_true")
    parser.add_argument("--trash-root", default=".trash/generated_artifacts")
    parser.add_argument("--confirm", default=None)
    args = parser.parse_args()

    rules = load_cleanup_rules(args.config)
    candidates = scan_generated_artifacts(args.root, rules, include_experiment_outputs=args.include_experiment_outputs)
    protected_count = len(rules.get("protected_paths", []))
    total_size = sum(c.size_bytes for c in candidates)

    # 中文注释：默认只 dry-run；正式执行也先移动到本地 trash，不要手动 rm -rf。
    action_report = {"executed": False, "dry_run": True}
    if args.permanent:
        action_report = delete_candidates_permanently(
            candidates,
            execute=args.execute,
            require_confirm_token=args.confirm,
            root=args.root,
        )
    elif args.execute:
        action_report = move_candidates_to_trash(candidates, args.trash_root, execute=True, root=args.root)
    else:
        action_report = move_candidates_to_trash(candidates, args.trash_root, execute=False, root=args.root)

    report = {
        "root": str(Path(args.root).resolve()),
        "candidates": candidates,
        "protected_count": protected_count,
        "dry_run": not args.execute,
        "executed": bool(args.execute),
        "permanent": bool(args.permanent),
        "action": action_report,
    }
    save_cleanup_report(report, args.out)

    print(f"candidate count: {len(candidates)}")
    print(f"protected path count: {protected_count}")
    print(f"total candidate size bytes: {total_size}")
    print(f"report saved to: {Path(args.out).resolve()}")
    preview = [c.path for c in candidates[:20]]
    compact_action = {
        "executed": bool(action_report.get("executed", False)),
        "trash_dir": action_report.get("trash_dir"),
        "num_candidates": action_report.get("num_candidates", len(candidates)),
        "preview_candidates": preview,
        "preview_truncated": len(candidates) > len(preview),
    }
    print(json.dumps(compact_action, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
