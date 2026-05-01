from __future__ import annotations

import torch

from shapesplat.editing.ops import apply_edit
from shapesplat.frontend.pipeline import FrontEndOutput
from shapesplat.gaussian.scene import ObjectGaussianScene
from shapesplat.geometry.masks import dilate_mask
from shapesplat.renderer.types import RenderOutput


def edited_scene(scene: ObjectGaussianScene, object_id: int, op: str) -> ObjectGaussianScene:
    """返回一个编辑后的 scene 副本，支持 remove/translate/scale/isolate。

    最小版本用 deepcopy 实现清晰语义；真实系统可改为更高效的参数视图或 batched edit renderer。
    """
    # optimization/edit_ops.py 保留训练 loss 旧接口；实际 buffer 编辑复用 inference editing/ops.py。
    return apply_edit(scene, {"op": op, "object_id": object_id, "translation": [0.18, 0.0, 0.0], "scale": 1.15})


def edit_consistency_loss(
    scene: ObjectGaussianScene,
    renderer,
    front: FrontEndOutput,
    base_render: RenderOutput,
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """可微编辑一致性。

    editability 不只是 separate buffer 的自然结果，而是通过 loss 显式约束：
    对某个 object 做 translate/remove 后，非编辑区域应尽量保持 RGB/alpha 不变，减少 collateral change。
    """
    if len(scene.objects) == 0:
        z = torch.tensor(0.0, device=front.image.device)
        return z, z
    obj_id = int(torch.randint(0, len(scene.objects), (1,), device=front.image.device).item())
    op = "translate" if obj_id % 2 == 0 else "remove"
    r_edit = renderer(edited_scene(scene, obj_id, op))
    edit_support = torch.maximum(front.masks[obj_id], base_render.ownership[obj_id].detach())
    edit_support = dilate_mask(edit_support, int(cfg["edit"]["dilate_radius"]))
    keep = 1.0 - edit_support
    denom = keep.sum().clamp_min(1)
    rgb_loss = ((r_edit.rgb - base_render.rgb.detach()).abs() * keep[None]).sum() / (denom * 3)
    alpha_loss = ((r_edit.alpha - base_render.alpha.detach()).abs() * keep).sum() / denom
    return rgb_loss, alpha_loss
