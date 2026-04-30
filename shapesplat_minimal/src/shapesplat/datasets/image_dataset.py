from __future__ import annotations

from pathlib import Path

from shapesplat.data.image_io import load_image
from shapesplat.datasets.manifest import ImageRecord, load_manifest


class ImageDataset:
    """最小 RGB image dataset。

    当前 dataset 只负责加载和 resize RGB 输入，不负责 GT mask、GT mesh 或相机标定。
    后续正式 benchmark 可以在 ImageRecord.metadata 中扩展这些字段。
    """

    def __init__(self, records: list[ImageRecord], image_size: int):
        self.records = records
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = load_image(record.image_path, size=self.image_size)
        return {
            "image_id": record.image_id,
            "image_path": record.image_path,
            "image": image,
            "record": record,
        }


def build_dataset_from_manifest(manifest_path: str | Path, image_size: int) -> ImageDataset:
    """从 manifest.csv 构建 ImageDataset。"""
    return ImageDataset(load_manifest(manifest_path), image_size=image_size)
