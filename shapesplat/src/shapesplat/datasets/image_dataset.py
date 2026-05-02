from __future__ import annotations

from pathlib import Path

from shapesplat.data.image_io import image_resize_kwargs_from_cfg, load_image
from shapesplat.datasets.manifest import ImageRecord, load_manifest


class ImageDataset:
    """最小 RGB image dataset。

    当前 dataset 只负责加载和 resize RGB 输入，不负责 GT mask、GT mesh 或相机标定。
    后续正式 benchmark 可以在 ImageRecord.metadata 中扩展这些字段。
    """

    def __init__(self, records: list[ImageRecord], image_size: int | None = None, image_cfg: dict | None = None):
        self.records = records
        self.image_size = int(image_size) if image_size is not None else None
        self.image_cfg = image_cfg or {"size": self.image_size, "resize_mode": "square"}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        full_image = load_image(record.image_path, resize_mode="none")
        image = load_image(record.image_path, **image_resize_kwargs_from_cfg({"image": self.image_cfg}))
        # 中文注释：CO3Dv2 high-res workflow 需要同时记录原始分辨率和 pipeline 工作分辨率。
        record.metadata["original_image_shape"] = list(full_image.shape)
        record.metadata["working_image_shape"] = list(image.shape)
        return {
            "image_id": record.image_id,
            "image_path": record.image_path,
            "image": image,
            "image_fullres": full_image,
            "record": record,
        }


def build_dataset_from_manifest(manifest_path: str | Path, image_size: int | None = None, cfg: dict | None = None) -> ImageDataset:
    """从 manifest.csv 构建 ImageDataset。"""
    image_cfg = cfg.get("image", {}) if cfg else {"size": image_size, "resize_mode": "square"}
    return ImageDataset(load_manifest(manifest_path), image_size=image_size, image_cfg=image_cfg)
