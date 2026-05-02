from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def set_by_dotted_key(cfg: dict, dotted_key: str, value: Any) -> None:
    """用 dotted key 修改嵌套配置。

    示例：set_by_dotted_key(cfg, "ablation.use_hidden_prior", False)。
    如果中间 key 不存在，会创建嵌套 dict；这让 ablation 文件可以覆盖未来新增字段。
    """
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            print(f"Warning: creating missing config section for override: {part}")
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """深拷贝 cfg 并应用多个 dotted-key overrides。"""
    out = deepcopy(cfg)
    for key, value in (overrides or {}).items():
        set_by_dotted_key(out, key, value)
    return out


def load_ablation_file(path: str | Path) -> list[dict]:
    """读取 configs/ablations.yaml，返回 experiments 列表。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ablation 配置不存在: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    exps = data.get("experiments", [])
    if not isinstance(exps, list) or len(exps) == 0:
        raise ValueError(f"ablation 配置中没有 experiments: {p}")
    return exps
