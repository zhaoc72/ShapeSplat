from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BaselineInputSpec:
    """baseline 共享输入协议。

    这个 spec 的核心目的是记录同一张图像和同一组 retained visible masks。
    后续 object-centric baseline、scene-level baseline 都应读取这些输入，避免
    proposal quality 差异污染 reconstruction / ownership / editing 对比。
    """

    image_id: str
    image_path: str
    masks_path: str
    output_dir: str
    num_objects: int
    crop_dir: str
    metadata_path: str


@dataclass
class BaselineOutputSpec:
    """baseline 输出协议。

    真实 baseline 后续只要写出 RGB、alpha、ownership 或 per-object alpha，
    就可以复用本项目的 2D/ownership 指标进行统一评估。
    """

    method_name: str
    image_id: str
    output_dir: str
    render_path: Optional[str]
    alpha_path: Optional[str]
    ownership_path: Optional[str]
    metrics_path: Optional[str]
    object_alpha_paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _write_dataclass(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(obj), f, indent=2, ensure_ascii=False)


def write_baseline_input_spec(spec: BaselineInputSpec, path: str | Path) -> None:
    """将 baseline 输入协议保存为 JSON。"""

    _write_dataclass(spec, path)


def read_baseline_input_spec(path: str | Path) -> BaselineInputSpec:
    """从 JSON 读取 baseline 输入协议。"""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BaselineInputSpec(**data)


def write_baseline_output_spec(spec: BaselineOutputSpec, path: str | Path) -> None:
    """将 baseline 输出协议保存为 JSON。"""

    _write_dataclass(spec, path)


def read_baseline_output_spec(path: str | Path) -> BaselineOutputSpec:
    """从 JSON 读取 baseline 输出协议。"""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BaselineOutputSpec(**data)

