from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageRecord:
    """一条 batch experiment 输入图像记录。

    manifest 用于后续正式 batch experiment：它只描述输入 RGB 图像和元信息，
    当前 minimal 版本不要求 GT mask / GT mesh。
    """

    image_id: str
    image_path: str
    split: str = "test"
    metadata: dict = field(default_factory=dict)


def load_manifest(path: str | Path) -> list[ImageRecord]:
    """读取 CSV manifest。

    CSV 至少需要 `image_id,image_path` 两列。额外列如 category/notes 会存入
    metadata。相对 image_path 会按 manifest 所在目录解析，方便数据集整体移动。
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path.resolve()}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest path is not a file: {manifest_path.resolve()}")

    records: list[ImageRecord] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        required = {"image_id", "image_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader):
            image_id = (row.get("image_id") or "").strip()
            image_path_raw = (row.get("image_path") or "").strip()
            if not image_id or not image_path_raw:
                raise ValueError(f"Manifest row {row_idx} has empty image_id or image_path.")
            image_path = Path(image_path_raw)
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            split = (row.get("split") or "test").strip() or "test"
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"image_id", "image_path", "split"} and value not in (None, "")
            }
            # mask_path 用于 same-mask setting。相对路径按 manifest 所在目录解析，
            # 这样 dataset 可以整体移动，且 batch runner 能直接传给 file mask loader。
            if "mask_path" in metadata:
                mask_path = Path(str(metadata["mask_path"]))
                if not mask_path.is_absolute():
                    mask_path = manifest_path.parent / mask_path
                metadata["mask_path"] = str(mask_path)
            if "metadata_path" in metadata:
                metadata_path = Path(str(metadata["metadata_path"]))
                if not metadata_path.is_absolute():
                    metadata_path = manifest_path.parent / metadata_path
                metadata["metadata_path"] = str(metadata_path)
            for cache_key in ("frontend_cache_dir", "cache_dir"):
                if cache_key in metadata:
                    cache_path = Path(str(metadata[cache_key]))
                    if not cache_path.is_absolute():
                        cache_path = manifest_path.parent / cache_path
                    # frontend_cache_dir 是后续 runner 读取 cached frontend outputs 的标准键。
                    metadata["frontend_cache_dir"] = str(cache_path)
            records.append(ImageRecord(image_id=image_id, image_path=str(image_path), split=split, metadata=metadata))

    if not records:
        raise ValueError(f"Manifest contains no records: {manifest_path}")
    return records
