"""Ours 主方法 benchmark runner。

这个包只编排已有 ShapeSplat++ 训练、渲染、评估和输出协议，不新增核心算法。
"""

from .ours_runner import run_ours_benchmark, run_ours_single, run_ours_variants_benchmark
from .readiness import check_ours_core_ready
from .variants import apply_variant_overrides, get_variant_by_name, load_ours_variants

__all__ = [
    "run_ours_single",
    "run_ours_benchmark",
    "run_ours_variants_benchmark",
    "check_ours_core_ready",
    "load_ours_variants",
    "get_variant_by_name",
    "apply_variant_overrides",
]
