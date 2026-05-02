import torch

from .camera import Camera


def project_points(camera: Camera, xyz: torch.Tensor) -> torch.Tensor:
    """投影工具函数，保留给后续 renderer/geometry 模块复用。"""
    return camera.project(xyz)
