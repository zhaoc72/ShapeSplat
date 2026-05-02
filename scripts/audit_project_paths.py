from __future__ import annotations

import argparse
import json
from pathlib import Path


OLD_PATTERNS = [
    r"C:\Users\zhaoc\ShapeSplat\shapesplat_minimal",
    r"C:\\Users\\zhaoc\\ShapeSplat\\shapesplat_minimal",
    "C:/Users/zhaoc/ShapeSplat/shapesplat_minimal",
    r"C:\Users\zhaoc\ShapeSplat\shapesplat",
    r"C:\\Users\\zhaoc\\ShapeSplat\\shapesplat",
    "C:/Users/zhaoc/ShapeSplat/shapesplat",
    "shapesplat_minimal",
]
NEW_ROOT = r"C:\Users\zhaoc\ShapeSplat"
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".json", ".csv", ".bat", ".ps1"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".trash"}


def _skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    normalized = str(path).replace("\\", "/")
    # 中文注释：审计脚本本身会保存旧路径匹配规则，历史审计/迁移报告也会引用旧路径；
    # 这些不是当前项目实际运行入口，避免自引用导致 after audit 永远不为零。
    if normalized.endswith("scripts/audit_project_paths.py"):
        return True
    if "/outputs/path_audit/" in normalized or "/outputs/codebase_audit/" in normalized:
        return True
    if "/outputs/project_root_migration/" in normalized:
        return True
    if "/outputs/test_" in normalized:
        return True
    if "/dist/" in normalized or ".egg-info/" in normalized:
        return True
    if "ShapeSplat_backup_before_flatten" in str(path):
        return True
    return False


def _is_text(path: Path) -> bool:
    try:
        return path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 2_000_000
    except OSError:
        return False


def _replacement(text: str) -> str:
    out = text
    replacements = {
        r"C:\Users\zhaoc\ShapeSplat\shapesplat_minimal": NEW_ROOT,
        r"C:\\Users\\zhaoc\\ShapeSplat\\shapesplat_minimal": r"C:\\Users\\zhaoc\\ShapeSplat",
        "C:/Users/zhaoc/ShapeSplat/shapesplat_minimal": "C:/Users/zhaoc/ShapeSplat",
        r"C:\Users\zhaoc\ShapeSplat\shapesplat": NEW_ROOT,
        r"C:\\Users\\zhaoc\\ShapeSplat\\shapesplat": r"C:\\Users\\zhaoc\\ShapeSplat",
        "C:/Users/zhaoc/ShapeSplat/shapesplat": "C:/Users/zhaoc/ShapeSplat",
        "dist/shapesplat_artifact.zip": "dist/shapesplat_artifact.zip",
        "shapesplat_artifact.zip": "shapesplat_artifact.zip",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def scan_paths(root: str | Path) -> list[dict]:
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if _skip(path) or not path.is_file() or not _is_text(path):
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, 1):
            hits = [p for p in OLD_PATTERNS if p in line]
            if hits:
                # 中文注释：agents.md 需要明确声明旧路径已废弃，因此这里保留记录但不视为待修复项。
                explanatory = rel == "agents.md"
                rows.append(
                    {
                        "file": rel,
                        "line": idx,
                        "hits": hits,
                        "snippet": line.strip()[:300],
                        "needs_fix": not explanatory,
                        "note": "deprecated path note" if explanatory else "old project path reference",
                    }
                )
    return rows


def replace_paths(root: str | Path) -> list[str]:
    root = Path(root).resolve()
    changed = []
    for path in sorted(root.rglob("*")):
        if _skip(path) or not path.is_file() or not _is_text(path):
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel == "agents.md":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        new_text = _replacement(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed.append(str(path.relative_to(root)).replace("\\", "/"))
    return changed


def _write_report(rows: list[dict], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    actionable = sum(1 for row in rows if row.get("needs_fix", True))
    lines = [f"# Project Path Audit {name}", "", f"- Matches: `{len(rows)}`", f"- Needs fix: `{actionable}`", ""]
    for row in rows:
        mark = "needs-fix" if row.get("needs_fix", True) else "informational"
        lines.append(f"- `{row['file']}:{row['line']}` [{mark}] {row['snippet']}")
    (out_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and optionally replace old ShapeSplat project root paths.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="outputs/path_audit")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    before = scan_paths(root)
    _write_report(before, out, "path_audit_before")
    changed = []
    if args.replace:
        # 中文注释：只改文本文件中的项目根目录引用，不碰 D:\projects 外部数据/模型路径。
        changed = replace_paths(root)
    after = scan_paths(root)
    _write_report(after, out, "path_audit_after")
    summary = {
        "before_count": len(before),
        "before_needs_fix": sum(1 for row in before if row.get("needs_fix", True)),
        "after_count": len(after),
        "after_needs_fix": sum(1 for row in after if row.get("needs_fix", True)),
        "changed_files": changed,
    }
    (out / "path_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
