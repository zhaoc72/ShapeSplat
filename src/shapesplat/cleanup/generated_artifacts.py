from __future__ import annotations

import csv
import json
import shutil
from fnmatch import fnmatch
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CleanupCandidate:
    path: str
    reason: str
    group: str
    size_bytes: int
    is_directory: bool
    protected: bool = False


def _resolve_root(root: str | Path) -> Path:
    return Path(root).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _protected_paths(root: Path, rules: dict) -> list[Path]:
    return [(root / p).resolve() for p in rules.get("protected_paths", [])]


def _is_protected(root: Path, path: Path, rules: dict) -> bool:
    resolved = path.resolve()
    if not _is_under(resolved, root):
        return True
    return any(resolved == protected or _is_under(resolved, protected) for protected in _protected_paths(root, rules))


def _is_default_keep(root: Path, path: Path, rules: dict) -> bool:
    rel = _relative(root, path)
    return any(fnmatch(rel, pattern) for pattern in rules.get("default_keep_globs", []))


def _matches_any(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch(rel, pattern) for pattern in patterns)


def _is_safe_empty_dir_candidate(root: Path, path: Path, rules: dict) -> bool:
    rel = _relative(root, path)
    patterns = rules.get("default_candidate_patterns", {})

    # 中文注释：空目录清理默认只覆盖明确的临时/调试区域。outputs 下的 final/paper/report
    # 等实验目录即使为空，也不应在默认清理里被列为候选，避免误伤正式结果结构。
    if rel.startswith("outputs/"):
        explicit_output_patterns = list(patterns.get("paths", [])) + list(patterns.get("globs", []))
        explicit_output_patterns += ["outputs/test_*", "outputs/test_*/**"]
        return _matches_any(rel, explicit_output_patterns)

    if rel.startswith(".trash/"):
        return False

    return True


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += int(child.stat().st_size)
        except OSError:
            pass
    return total


def _add_candidate(candidates: dict[str, CleanupCandidate], root: Path, path: Path, reason: str, group: str, rules: dict) -> None:
    if not path.exists():
        return
    protected = _is_protected(root, path, rules)
    if protected:
        return
    if group != "experiment_output" and _is_default_keep(root, path, rules):
        return
    rel = _relative(root, path)
    candidates[rel] = CleanupCandidate(
        path=rel,
        reason=reason,
        group=group,
        size_bytes=_path_size(path),
        is_directory=path.is_dir(),
        protected=False,
    )


def _iter_existing_globs(root: Path, patterns: list[str]):
    for pattern in patterns:
        yield from root.glob(pattern)


def _scan_empty_dirs(root: Path, rules: dict, candidates: dict[str, CleanupCandidate]) -> None:
    # 中文注释：空目录只作为低风险候选，但源码/配置/数据保护树下的空目录不会返回。
    dirs = sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in dirs:
        if d == root or _is_protected(root, d, rules):
            continue
        if not _is_safe_empty_dir_candidate(root, d, rules):
            continue
        try:
            if not any(d.iterdir()):
                _add_candidate(candidates, root, d, "empty directory", "empty_dir", rules)
        except OSError:
            continue


def scan_generated_artifacts(root: str | Path, rules: dict, include_experiment_outputs: bool = False) -> list[CleanupCandidate]:
    """扫描测试/调试生成文件。

    中文注释：该函数只返回未受保护的候选；源码、配置、测试、文档、CO3Dv2 benchmark
    和真实 cache 即使匹配某些模式也不会成为可执行清理项。
    """
    root_path = _resolve_root(root)
    candidates: dict[str, CleanupCandidate] = {}
    patterns = rules.get("default_candidate_patterns", {})

    for p in root_path.rglob("*"):
        if _is_protected(root_path, p, rules):
            continue
        if p.is_dir():
            if p.name in patterns.get("dir_names", []):
                _add_candidate(candidates, root_path, p, f"directory name {p.name}", "python_cache" if "pytest" in p.name or p.name == "__pycache__" else "build_cache", rules)
            if any(p.name.endswith(suffix) for suffix in patterns.get("dir_suffixes", [])):
                _add_candidate(candidates, root_path, p, f"directory suffix {p.name}", "build_cache", rules)
        elif p.is_file():
            if p.name in patterns.get("file_names", []):
                _add_candidate(candidates, root_path, p, f"file name {p.name}", "build_cache", rules)
            if any(p.name.endswith(suffix) for suffix in patterns.get("file_suffixes", [])):
                _add_candidate(candidates, root_path, p, f"file suffix {p.suffix}", "python_cache", rules)

    for rel in patterns.get("paths", []):
        _add_candidate(candidates, root_path, root_path / rel, f"explicit generated path {rel}", "debug_output", rules)
    for p in _iter_existing_globs(root_path, patterns.get("globs", [])):
        group = "dist" if p.as_posix().endswith(".zip") else "debug_output"
        _add_candidate(candidates, root_path, p, f"generated glob match {p.name}", group, rules)

    if include_experiment_outputs:
        optional = rules.get("optional_experiment_output_patterns", {})
        for p in _iter_existing_globs(root_path, optional.get("globs", [])):
            _add_candidate(candidates, root_path, p, f"optional experiment output {p.name}", "experiment_output", rules)

    _scan_empty_dirs(root_path, rules, candidates)
    return sorted(candidates.values(), key=lambda c: c.path)


def move_candidates_to_trash(candidates, trash_root: str | Path, execute: bool = False, root: str | Path | None = None) -> dict:
    """把候选移动到本地 trash；execute=False 时只模拟。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    trash = Path(trash_root)
    if not trash.is_absolute():
        trash = root / trash
    trash = trash / timestamp
    moved = []
    for candidate in candidates:
        src = root / candidate.path
        dst = trash / candidate.path
        moved.append({"src": str(src), "dst": str(dst), "executed": bool(execute)})
        if execute:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst))
    return {"trash_dir": str(trash), "num_candidates": len(candidates), "mappings": moved, "executed": bool(execute)}


def delete_candidates_permanently(
    candidates,
    execute: bool = False,
    require_confirm_token: str | None = None,
    root: str | Path | None = None,
):
    """永久删除候选；必须显式确认，默认拒绝。

    中文注释：项目默认推荐 move-to-trash，不推荐永久删除；该接口只给高级用户兜底。
    """
    if require_confirm_token != "DELETE_GENERATED_ARTIFACTS":
        raise ValueError("Permanent deletion requires --confirm DELETE_GENERATED_ARTIFACTS")
    deleted = []
    root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    for candidate in candidates:
        path = root / candidate.path
        deleted.append(str(path))
        if execute and path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return {"executed": bool(execute), "num_deleted": len(deleted), "deleted": deleted}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_cleanup_report(report: dict, out_dir: str | Path) -> None:
    """保存清理候选和摘要报告。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = report.get("candidates", [])
    rows = [asdict(c) if isinstance(c, CleanupCandidate) else dict(c) for c in candidates]
    summary = {
        "dry_run": bool(report.get("dry_run", True)),
        "executed": bool(report.get("executed", False)),
        "permanent": bool(report.get("permanent", False)),
        "num_candidates": len(rows),
        "total_size_bytes": int(sum(int(r.get("size_bytes", 0)) for r in rows)),
        "protected_count": int(report.get("protected_count", 0)),
    }
    (out / "cleanup_candidates.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out / "cleanup_candidates.csv", rows)
    (out / "cleanup_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Generated Artifact Cleanup Summary",
        "",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Executed: `{summary['executed']}`",
        f"- Permanent: `{summary['permanent']}`",
        f"- Candidates: `{summary['num_candidates']}`",
        f"- Total size bytes: `{summary['total_size_bytes']}`",
        f"- Protected paths count: `{summary['protected_count']}`",
        "",
        "Cleanup defaults are conservative. Source, configs, tests, docs, CO3Dv2 benchmark, and real frontend caches are protected.",
    ]
    (out / "cleanup_summary.md").write_text("\n".join(lines), encoding="utf-8")
