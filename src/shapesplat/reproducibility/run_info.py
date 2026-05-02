from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class RunInfo:
    """一次实验的最小身份证。

    run_info 记录 run_id、命令、配置、输入和输出目录，便于之后复现实验。
    """

    run_id: str
    run_type: str
    timestamp: str
    status: str
    output_dir: str
    command: str
    config_path: Optional[str]
    input_path: Optional[str]
    manifest_path: Optional[str]
    seed: Optional[int]
    notes: dict


def generate_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def create_run_info(
    run_type: str,
    output_dir: str | Path,
    command: str | None = None,
    config_path: str | None = None,
    input_path: str | None = None,
    manifest_path: str | None = None,
    seed: int | None = None,
    notes: dict | None = None,
    status: str = "success",
) -> RunInfo:
    return RunInfo(
        run_id=generate_run_id(run_type),
        run_type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        status=status,
        output_dir=str(output_dir),
        command=command or "",
        config_path=str(config_path) if config_path is not None else None,
        input_path=str(input_path) if input_path is not None else None,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        seed=seed,
        notes=notes or {},
    )


def save_run_info(run_info: RunInfo, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(run_info), f, indent=2, ensure_ascii=False)


def load_run_info(path: str | Path) -> RunInfo:
    with open(path, "r", encoding="utf-8") as f:
        return RunInfo(**json.load(f))
