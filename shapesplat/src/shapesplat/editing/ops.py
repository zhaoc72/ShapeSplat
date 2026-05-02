from __future__ import annotations

import copy
import math

from shapesplat.gaussian.scene import ObjectGaussianScene


def clone_scene(scene: ObjectGaussianScene) -> ObjectGaussianScene:
    """深拷贝 scene，保证推理阶段编辑不会污染原始 Gaussian buffers。"""

    return copy.deepcopy(scene)


def _find_object(scene: ObjectGaussianScene, object_id: int):
    for obj in scene.objects:
        if int(obj.object_id) == int(object_id):
            return obj
    raise IndexError(f"object_id not found: {object_id}")


def remove_object(scene: ObjectGaussianScene, object_id: int) -> ObjectGaussianScene:
    """删除物体：在 object buffer 层把该物体 opacity logits 置为极小值。"""

    edited = clone_scene(scene)
    _find_object(edited, object_id).opacity_logits.data.fill_(-20.0)
    return edited


def translate_object(scene: ObjectGaussianScene, object_id: int, translation: tuple[float, float, float]) -> ObjectGaussianScene:
    """平移物体：直接平移指定 object 的 Gaussian means，不做图像后处理。"""

    edited = clone_scene(scene)
    obj = _find_object(edited, object_id)
    delta = obj.means.new_tensor(translation).view(1, 3)
    obj.means.data.add_(delta)
    return edited


def scale_object(scene: ObjectGaussianScene, object_id: int, scale: float) -> ObjectGaussianScene:
    """围绕 object center 缩放 Gaussian means，并同步调整 log_scales。"""

    edited = clone_scene(scene)
    obj = _find_object(edited, object_id)
    s = float(scale)
    center = obj.means.data.mean(dim=0, keepdim=True)
    obj.means.data[:] = center + (obj.means.data - center) * s
    obj.log_scales.data.add_(math.log(max(s, 1e-6)))
    return edited


def isolate_object(scene: ObjectGaussianScene, object_id: int) -> ObjectGaussianScene:
    """隔离物体：保留目标 object，压低其他 objects 的 opacity。"""

    edited = clone_scene(scene)
    for obj in edited.objects:
        if int(obj.object_id) != int(object_id):
            obj.opacity_logits.data.fill_(-20.0)
    return edited


def render_object_only(scene: ObjectGaussianScene, object_id: int) -> ObjectGaussianScene:
    """语义化别名：用于单独渲染 object asset，行为等同 isolate。"""

    return isolate_object(scene, object_id)


def apply_edit(scene: ObjectGaussianScene, edit_spec: dict) -> ObjectGaussianScene:
    """根据 edit_spec 分发 object-level edit operation。"""

    op = edit_spec.get("op", "remove")
    object_id = int(edit_spec.get("object_id", 0))
    if op == "remove":
        return remove_object(scene, object_id)
    if op == "translate":
        return translate_object(scene, object_id, tuple(edit_spec.get("translation", [0.12, 0.0, 0.0])))
    if op == "scale":
        return scale_object(scene, object_id, float(edit_spec.get("scale", 1.2)))
    if op == "isolate":
        return isolate_object(scene, object_id)
    if op in {"object_only", "render_object_only"}:
        return render_object_only(scene, object_id)
    raise ValueError(f"Unknown edit op: {op}")

