"""Shape bank 与 hidden support prior 模块。

默认使用 ToyShapeBank；v0.8 起可以通过 FileShapeBank 读取本地 .npz/.npy 点云。
"""

from .types import ShapeAsset
from .toy_shape_bank import ToyShapeBank
from .file_shape_bank import FileShapeBank
from .shape_bank_backend import build_shape_bank

__all__ = ["ShapeAsset", "ToyShapeBank", "FileShapeBank", "build_shape_bank"]
