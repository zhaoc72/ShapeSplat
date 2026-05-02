from __future__ import annotations

import torch


def make_identity_rotations(num: int, device: torch.device) -> torch.Tensor:
    """生成真实 3DGS renderer 常用的 identity quaternion。
    当前内部 GaussianObject 还没有旋转参数，因此用 [1,0,0,0] 作为兼容占位。
    """

    rotations = torch.zeros((int(num), 4), device=device, dtype=torch.float32)
    if num > 0:
        rotations[:, 0] = 1.0
    return rotations


def extract_gaussian_tensors(scene) -> dict:
    """把 ShapeSplat++ 内部 object Gaussian buffer 转成真实 renderer 常见张量。

    这是内部表示和 diff-gaussian / gsplat / custom CUDA renderer 之间的桥：
    - scales 使用 exp(log_scales)；
    - opacities 使用 sigmoid(opacity_logits)；
    - colors 使用 sigmoid(colors)；
    - rotations 目前用 identity quaternion，后续真实 3DGS 可替换为可学习旋转。
    """

    means, scales, opacities, colors, object_ids, branch_ids = [], [], [], [], [], []
    for obj in scene.objects:
        k = int(obj.means.shape[0])
        means.append(obj.means.float())
        scales.append(obj.log_scales.float().exp())
        opacities.append(obj.opacity_logits.float().sigmoid().reshape(k, 1))
        colors.append(obj.colors.float().sigmoid())
        object_ids.append(torch.full((k,), int(obj.object_id), device=obj.means.device, dtype=torch.long))
        branch_ids.append(obj.branch_ids.to(device=obj.means.device, dtype=torch.long))

    if not means:
        device = torch.device("cpu")
        empty = torch.empty((0,), device=device)
        return {
            "means": torch.empty((0, 3), device=device),
            "scales": torch.empty((0, 3), device=device),
            "rotations": torch.empty((0, 4), device=device),
            "opacities": torch.empty((0, 1), device=device),
            "colors": torch.empty((0, 3), device=device),
            "object_ids": empty.long(),
            "branch_ids": empty.long(),
        }

    means_t = torch.cat(means, dim=0)
    return {
        "means": means_t,
        "scales": torch.cat(scales, dim=0),
        "rotations": make_identity_rotations(means_t.shape[0], means_t.device),
        "opacities": torch.cat(opacities, dim=0),
        "colors": torch.cat(colors, dim=0),
        "object_ids": torch.cat(object_ids, dim=0),
        "branch_ids": torch.cat(branch_ids, dim=0),
    }


def filter_gaussians_by_object(tensors: dict, object_id: int) -> dict:
    """从已展开的 Gaussian tensors 中筛选单个 object。

    object-wise rendering fallback 会用这个函数逐物体渲染 alpha_n，
    从而在真实 renderer 没有 native contribution 时近似 ownership maps。
    """

    keep = tensors["object_ids"] == int(object_id)
    out = {}
    for key, value in tensors.items():
        out[key] = value[keep] if torch.is_tensor(value) and value.shape[:1] == keep.shape else value
    return out
