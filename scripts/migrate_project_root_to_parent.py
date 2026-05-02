from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_SOURCE = Path(r"C:\Users\zhaoc\ShapeSplat")
DEFAULT_TARGET = Path(r"C:\Users\zhaoc\ShapeSplat")
DEFAULT_BACKUP = Path(r"C:\Users\zhaoc\ShapeSplat_backup_before_flatten")
GENERATED_SKIP_NAMES = {".pytest_basetemp", ".pytest_cache", ".pytest_tmp", "__pycache__"}


def _bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _project_markers_ok(source: Path) -> tuple[bool, list[str]]:
    required = ["pyproject.toml", "src", "scripts", "configs"]
    missing = [name for name in required if not (source / name).exists()]
    return len(missing) == 0, missing


def _top_level_entries(source: Path) -> list[Path]:
    # 中文注释：pytest 临时目录可能被 Windows 锁定或拒绝访问；它们不是项目内容，迁移时跳过。
    return sorted([p for p in source.iterdir() if p.name not in GENERATED_SKIP_NAMES], key=lambda p: p.name.lower())


def _ignore_generated(dir_path: str, names: list[str]) -> set[str]:
    return {name for name in names if name in GENERATED_SKIP_NAMES or name == "__pycache__"}


def _make_plan(source: Path, target: Path, backup_root: Path, overwrite: bool, remove_empty_source: bool) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root.parent / f"{backup_root.name}_{timestamp}"
    entries = _top_level_entries(source) if source.exists() else []
    conflicts = []
    moves = []
    for entry in entries:
        dst = target / entry.name
        if dst.exists():
            conflicts.append({"name": entry.name, "source": str(entry), "target": str(dst)})
        moves.append({"source": str(entry), "target": str(dst), "is_dir": entry.is_dir()})
    return {
        "source": str(source),
        "target": str(target),
        "backup_dir": str(backup_dir),
        "overwrite": bool(overwrite),
        "remove_empty_source": bool(remove_empty_source),
        "moves": moves,
        "conflicts": conflicts,
        "can_execute": len(conflicts) == 0 or bool(overwrite),
    }


def _write_report(plan: dict, report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "migration_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "migration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Project Root Migration Report",
        "",
        f"- Source: `{plan['source']}`",
        f"- Target: `{plan['target']}`",
        f"- Backup: `{plan['backup_dir']}`",
        f"- Executed: `{report.get('executed')}`",
        f"- Status: `{report.get('status')}`",
        f"- Conflicts: `{len(plan.get('conflicts', []))}`",
        "",
        "## Conflicts",
    ]
    if plan.get("conflicts"):
        lines.extend([f"- `{c['name']}`: `{c['target']}`" for c in plan["conflicts"]])
    else:
        lines.append("- None")
    lines += ["", "## Moves"]
    lines.extend([f"- `{m['source']}` -> `{m['target']}`" for m in plan.get("moves", [])])
    if report.get("notes"):
        lines += ["", "## Notes", *[f"- {n}" for n in report["notes"]]]
    (out_dir / "migration_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_migration(
    source: str | Path,
    target: str | Path,
    backup_root: str | Path,
    report_out: str | Path,
    dry_run: bool = True,
    execute: bool = False,
    overwrite: bool = False,
    remove_empty_source: bool = False,
) -> dict:
    source = Path(source).resolve()
    target = Path(target).resolve()
    backup_root = Path(backup_root).resolve()
    report_dir = Path(report_out)
    if not report_dir.is_absolute():
        report_dir = source / report_dir

    notes: list[str] = []
    errors: list[str] = []
    if not source.exists():
        errors.append(f"source does not exist: {source}")
    ok, missing = _project_markers_ok(source)
    if not ok:
        errors.append(f"source is missing project markers: {missing}")
    if not target.exists():
        errors.append(f"target does not exist: {target}")
    if source.parent.resolve() != target:
        errors.append("target must be the source parent directory")
    if (source / ".git").exists() and (target / ".git").exists():
        errors.append("both source and target contain .git; abort to avoid repository overwrite")

    plan = _make_plan(source, target, backup_root, overwrite, remove_empty_source) if not errors else {
        "source": str(source),
        "target": str(target),
        "backup_dir": "",
        "overwrite": bool(overwrite),
        "remove_empty_source": bool(remove_empty_source),
        "moves": [],
        "conflicts": [],
        "can_execute": False,
    }
    if plan.get("conflicts") and not overwrite:
        errors.append("target has conflicts; rerun with --overwrite true only after reviewing dry-run report")
    if dry_run or not execute:
        report = {"executed": False, "status": "dry_run_blocked" if errors else "dry_run_ok", "errors": errors, "notes": notes, "plan": plan}
        _write_report(plan, report, report_dir)
        return report
    if errors:
        report = {"executed": False, "status": "blocked", "errors": errors, "notes": notes, "plan": plan}
        _write_report(plan, report, report_dir)
        return report

    backup_dir = Path(plan["backup_dir"])
    backup_source = backup_dir / "source"
    backup_conflicts = backup_dir / "target_conflicts"
    backup_dir.mkdir(parents=True, exist_ok=False)
    # 中文注释：先完整备份 source，成功后才迁移；target 冲突项也搬到 backup，避免覆盖或删除。
    shutil.copytree(source, backup_source, dirs_exist_ok=False, ignore=_ignore_generated)
    notes.append(f"source backup created: {backup_source}")
    for conflict in plan.get("conflicts", []):
        target_conflict = Path(conflict["target"])
        dst = backup_conflicts / target_conflict.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target_conflict), str(dst))
        notes.append(f"target conflict moved to backup: {target_conflict} -> {dst}")

    moved = []
    for item in plan.get("moves", []):
        src = Path(item["source"])
        dst = Path(item["target"])
        shutil.move(str(src), str(dst))
        moved.append({"source": str(src), "target": str(dst)})

    source_removed = False
    if remove_empty_source:
        try:
            if source.exists() and not any(source.iterdir()):
                source.rmdir()
                source_removed = True
        except OSError as exc:
            notes.append(f"source not removed because it is not empty or locked: {exc}")

    # 报告目录随 source/outputs 移走后，重新定位到 target。
    final_report_dir = target / "outputs" / "project_root_migration"
    report = {
        "executed": True,
        "status": "success",
        "errors": [],
        "notes": notes,
        "plan": plan,
        "moved": moved,
        "backup_dir": str(backup_dir),
        "source_exists_after": source.exists(),
        "source_removed": source_removed,
    }
    _write_report(plan, report, final_report_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely flatten ShapeSplat project root from nested shapesplat/ to parent.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUP))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", default="false")
    parser.add_argument("--remove-empty-source", action="store_true")
    parser.add_argument("--report-out", default="outputs/project_root_migration")
    args = parser.parse_args()
    report = run_migration(
        args.source,
        args.target,
        args.backup_root,
        args.report_out,
        dry_run=(args.dry_run or not args.execute),
        execute=args.execute,
        overwrite=_bool_arg(args.overwrite),
        remove_empty_source=args.remove_empty_source,
    )
    print(json.dumps({"status": report.get("status"), "executed": report.get("executed"), "errors": report.get("errors", []), "backup_dir": report.get("backup_dir") or report.get("plan", {}).get("backup_dir")}, indent=2, ensure_ascii=False))
    if report.get("errors") and args.execute:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
