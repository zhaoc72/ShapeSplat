from __future__ import annotations

import json
import zipfile
from pathlib import Path


EXCLUDED_DIRS = {".git", ".venv", "outputs", "runs", "__pycache__", ".pytest_cache", "dist"}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt"}


def _should_include(path: Path, include_docs: bool, include_tests: bool, include_examples: bool) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIRS:
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if "docs" in parts and not include_docs:
        return False
    if "tests" in parts and not include_tests:
        return False
    if "examples" in parts and not include_examples:
        return False
    return path.is_file()


def make_artifact_manifest(root: str | Path = ".") -> dict:
    """生成 artifact 文件清单，不包含模型权重和大输出。"""

    root = Path(root)
    files = []
    total_size = 0
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if _should_include(rel, True, True, True):
            size = path.stat().st_size
            files.append({"path": str(rel).replace("\\", "/"), "size": size, "suffix": path.suffix})
            total_size += size
    return {
        "num_files": len(files),
        "estimated_size": total_size,
        "files": files,
        "excluded_patterns": sorted([*EXCLUDED_DIRS, *EXCLUDED_SUFFIXES]),
    }


def create_artifact_package(
    out_path: str | Path,
    include_docs: bool = True,
    include_tests: bool = True,
    include_examples: bool = True,
) -> Path:
    """创建轻量 artifact zip。

    该包排除 outputs/runs/.git/checkpoints 等大文件或运行产物；真实模型
    权重需要用户按 backend 文档单独配置。
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(".")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            rel = path.relative_to(root)
            if not _should_include(rel, include_docs, include_tests, include_examples):
                continue
            # 避免把正在生成的 zip 自己打进去。
            if path.resolve() == out_path.resolve():
                continue
            zf.write(path, rel.as_posix())
    manifest = make_artifact_manifest(root)
    (out_path.parent / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
