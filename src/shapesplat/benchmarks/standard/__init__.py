"""Standard same-mask benchmark helpers.

这些工具只验证/整理 retained visible masks，不生成新 proposal，也不引入真实 3D GT。
"""

from .builder import build_from_folder, build_same_mask_benchmark
from .validator import save_validation_report, validate_benchmark_manifest

__all__ = ["build_from_folder", "build_same_mask_benchmark", "save_validation_report", "validate_benchmark_manifest"]
