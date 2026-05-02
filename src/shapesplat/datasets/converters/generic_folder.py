from __future__ import annotations

from pathlib import Path

from shapesplat.datasets.benchmark.builder_v2 import build_benchmark_from_folders
from shapesplat.datasets.converters.base import DatasetConverter


class GenericFolderConverter(DatasetConverter):
    """通用文件夹 converter。

    支持 src/images、src/masks、可选 src/metadata；按同名文件匹配。
    """

    name = "generic_folder"

    def convert(self, src: str | Path, out: str | Path, cfg: dict | None = None) -> Path:
        cfg = cfg or {}
        src = Path(src)
        return build_benchmark_from_folders(
            src / "images",
            src / "masks",
            out,
            metadata_dir=src / "metadata" if (src / "metadata").exists() else None,
            source_dataset=cfg.get("source_dataset", "custom_folder"),
            split=cfg.get("split", "test"),
            overwrite=bool(cfg.get("overwrite", False)),
        )

