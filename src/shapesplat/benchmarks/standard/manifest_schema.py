from __future__ import annotations

from pathlib import Path

REQUIRED_COLUMNS = ["image_id", "image_path", "mask_path", "split"]
OPTIONAL_COLUMNS = [
    "metadata_path",
    "subset",
    "category",
    "num_objects",
    "scene_id",
    "camera_path",
    "depth_path",
    "mesh_path",
    "pointcloud_path",
]


def normalize_manifest_row(row: dict, manifest_dir: Path) -> dict:
    """规范化 same-mask benchmark manifest 行。

    标准格式要求所有方法共享同一张 image 和同一组 retained visible instance masks。
    """
    missing = [key for key in REQUIRED_COLUMNS if not str(row.get(key, "")).strip()]
    if missing:
        raise ValueError(f"manifest row missing required columns: {missing}")
    out = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
    for key in ["image_path", "mask_path", "metadata_path", "camera_path", "depth_path", "mesh_path", "pointcloud_path"]:
        value = out.get(key)
        if value:
            p = Path(str(value))
            out[key] = str(p if p.is_absolute() else (manifest_dir / p).resolve())
    return out
