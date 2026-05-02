from __future__ import annotations

import torch


def pinhole_to_fov(camera) -> tuple[float, float]:
    """把 canonical pinhole camera 转成真实 3DGS 常用的 tan_fovx / tan_fovy。"""

    tan_fovx = float(camera.width) / (2.0 * float(camera.fx))
    tan_fovy = float(camera.height) / (2.0 * float(camera.fy))
    return tan_fovx, tan_fovy


def make_identity_view_matrix(device) -> torch.Tensor:
    """当前 minimal camera 没有真实外参，因此 view matrix 使用 identity。"""

    return torch.eye(4, device=device, dtype=torch.float32)


def make_projection_matrix(camera, near: float, far: float) -> torch.Tensor:
    """构造一个轻量 perspective projection matrix。

    后续真实数据集可以替换为 COLMAP/真实相机矩阵；这里仅用于 renderer adapter
    兼容层和 smoke test，不改变现有 soft renderer 行为。
    """

    device = camera.device
    tan_fovx, tan_fovy = pinhole_to_fov(camera)
    proj = torch.zeros((4, 4), device=device, dtype=torch.float32)
    proj[0, 0] = 1.0 / max(tan_fovx, 1.0e-8)
    proj[1, 1] = 1.0 / max(tan_fovy, 1.0e-8)
    proj[2, 2] = float(far) / max(float(far) - float(near), 1.0e-8)
    proj[2, 3] = -(float(far) * float(near)) / max(float(far) - float(near), 1.0e-8)
    proj[3, 2] = 1.0
    return proj


def make_real_renderer_camera(camera, cfg: dict) -> dict:
    """生成真实 renderer adapter 使用的 camera 参数字典。

    当前 ShapeSplat++ minimal pipeline 使用 canonical pinhole camera；
    真实 benchmark 后续可以把 intrinsics/extrinsics 替换为真实相机协议。
    """

    rcfg = cfg.get("renderer", {}).get("real_3dgs", {})
    near = float(rcfg.get("near", cfg.get("camera", {}).get("z_near", 0.01)))
    far = float(rcfg.get("far", cfg.get("camera", {}).get("z_far", 100.0)))
    tan_fovx, tan_fovy = pinhole_to_fov(camera)
    bg = torch.tensor(rcfg.get("background_color", [1.0, 1.0, 1.0]), device=camera.device, dtype=torch.float32)
    return {
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "tan_fovx": tan_fovx,
        "tan_fovy": tan_fovy,
        "view_matrix": make_identity_view_matrix(camera.device),
        "projection_matrix": make_projection_matrix(camera, near, far),
        "near": near,
        "far": far,
        "background_color": bg,
    }
