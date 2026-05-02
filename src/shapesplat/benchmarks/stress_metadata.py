from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ObjectInfo:
    """stress benchmark 中单个 2D 前景物体的弱 GT 信息。"""

    object_id: int
    category: str
    color: list[int]
    bbox_xyxy: list[int]
    visible_area: int
    full_area_approx: int
    is_truncated: bool = False
    scale: float = 1.0


@dataclass
class StressMetadata:
    """synthetic stress benchmark 的诊断 metadata。

    这些字段用于定位遮挡、同类混淆和截断 failure mode，不是完整 3D GT。
    """

    image_id: str
    subset: str
    num_objects: int
    object_infos: list[ObjectInfo]
    occlusion_pairs: list[list[int]]
    depth_order_pairs: list[list[int]]
    same_category_pairs: list[list[int]]
    truncation_flags: dict[str, bool]
    scale_ratios: dict[str, float]
    generation_seed: int


def save_stress_metadata(meta: StressMetadata, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2, ensure_ascii=False)


def load_stress_metadata(path: str | Path) -> StressMetadata:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    infos = [ObjectInfo(**item) for item in data.get("object_infos", [])]
    data["object_infos"] = infos
    return StressMetadata(**data)

