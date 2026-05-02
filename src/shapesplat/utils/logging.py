import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path) -> None:
    """保存 json 日志，训练脚本用它写 loss_log.json。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
