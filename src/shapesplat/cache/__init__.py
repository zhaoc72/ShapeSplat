"""Frontend cache utilities.

缓存模块用于把真实 SAM / DINO / Depth 的前端输出落盘，避免批量实验时重复推理。
"""

from .frontend_cache import FrontendCacheRecord, frontend_cache_exists, load_frontend_output, save_frontend_output

__all__ = [
    "FrontendCacheRecord",
    "frontend_cache_exists",
    "load_frontend_output",
    "save_frontend_output",
]
