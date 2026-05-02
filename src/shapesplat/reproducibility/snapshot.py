from __future__ import annotations

import json
from pathlib import Path

import yaml


RESULT_SUFFIXES = {".png", ".json", ".csv", ".yaml", ".yml", ".npy", ".pt", ".md", ".tex", ".txt"}


def save_resolved_config(cfg: dict, out_path: str | Path) -> None:
    """保存完整 resolved config，优先 yaml。"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def save_command(command: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(command or "", encoding="utf-8")


def index_output_files(root: str | Path) -> dict:
    """扫描输出目录，记录常见结果文件。"""

    root = Path(root)
    files = []
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in RESULT_SUFFIXES:
                files.append({"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "suffix": p.suffix.lower()})
    return {"num_files": len(files), "files": files}


def save_output_index(root: str | Path, out_path: str | Path) -> dict:
    index = index_output_files(root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    return index

