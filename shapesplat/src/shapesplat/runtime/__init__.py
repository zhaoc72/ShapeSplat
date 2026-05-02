"""Windows / CUDA runtime helpers for ShapeSplat++.

中文注释：runtime 包只负责设备解析、CUDA 诊断、显存记录和 AMP 配置，
不改变 ShapeSplat++ 的算法逻辑。
"""

from .device import DeviceInfo, get_torch_device, resolve_runtime_device

__all__ = ["DeviceInfo", "get_torch_device", "resolve_runtime_device"]
