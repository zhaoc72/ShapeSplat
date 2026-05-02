from __future__ import annotations

from pathlib import Path


class DatasetConverter:
    """数据集 converter 基类。

    所有 converter 的输出都必须是 benchmark v2 manifest；原始数据解析可以因数据集
    本地格式而异，但最终协议保持一致。
    """

    name: str = "base"

    def convert(self, src: str | Path, out: str | Path, cfg: dict | None = None) -> Path:
        raise NotImplementedError

