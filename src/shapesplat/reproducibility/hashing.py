from __future__ import annotations

import hashlib
from pathlib import Path


DEFAULT_PATTERNS = [".json", ".yaml", ".yml", ".csv", ".png", ".npy", ".txt", ".md", ".tex"]


def hash_file(path: str | Path, algorithm: str = "sha256") -> str:
    """计算文件 hash，用于检查结果是否被修改。"""

    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(root: str | Path, patterns: list[str] | None = None, max_file_size_mb: int = 100) -> dict:
    """遍历输出目录并 hash 常见结果文件；默认跳过大 checkpoint。"""

    root = Path(root)
    suffixes = set(patterns or DEFAULT_PATTERNS)
    max_size = int(max_file_size_mb) * 1024 * 1024
    out = {}
    if not root.exists():
        return out
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in suffixes:
            continue
        size = path.stat().st_size
        if size > max_size:
            continue
        rel = path.relative_to(root).as_posix()
        out[rel] = {"sha256": hash_file(path), "size": size}
    return out


def write_file_hashes(root: str | Path, path: str | Path) -> dict:
    import json

    hashes = hash_directory(root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)
    return hashes

