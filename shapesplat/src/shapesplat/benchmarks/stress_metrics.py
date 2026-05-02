from __future__ import annotations

import math
from pathlib import Path

import torch

from shapesplat.benchmarks.stress_metadata import StressMetadata, load_stress_metadata


def load_metadata_for_record(record) -> StressMetadata | None:
    """从 manifest record 读取 stress metadata；缺失时返回 None，便于普通 dataset 复用。"""

    path = getattr(record, "metadata", {}).get("metadata_path") if record is not None else None
    if not path:
        return None
    path = Path(path)
    return load_stress_metadata(path) if path.exists() else None


def _argmax_ownership(ownership: torch.Tensor) -> torch.Tensor:
    if ownership.numel() == 0:
        return torch.empty(ownership.shape[-2:], dtype=torch.long, device=ownership.device)
    return ownership.argmax(dim=0)


def _purity_for_object(argmax: torch.Tensor, mask: torch.Tensor, object_id: int) -> float:
    valid = mask > 0.5
    if valid.sum() == 0:
        return 0.0
    return float((argmax[valid] == int(object_id)).float().mean().detach().cpu())


def swap_rate_proxy(ownership: torch.Tensor, masks: torch.Tensor, metadata: StressMetadata) -> float | None:
    """同类物体 identity swap 的轻量 proxy，不是严格 3D identity metric。"""

    pairs = metadata.same_category_pairs
    if not pairs:
        return None
    argmax = _argmax_ownership(ownership)
    swapped = 0
    total = 0
    for i, j in pairs:
        if i >= masks.shape[0] or j >= masks.shape[0] or i >= ownership.shape[0] or j >= ownership.shape[0]:
            continue
        mi, mj = masks[i] > 0.5, masks[j] > 0.5
        if mi.sum() == 0 or mj.sum() == 0:
            continue
        pred_i = int(torch.mode(argmax[mi]).values.item())
        pred_j = int(torch.mode(argmax[mj]).values.item())
        swapped += int(pred_i == j and pred_j == i)
        total += 1
    return float(swapped / total) if total else None


def order_accuracy_proxy(render_depth: torch.Tensor, masks: torch.Tensor, metadata: StressMetadata) -> float | None:
    """用全局 render depth 在 visible mask 内的 median 近似前后顺序诊断。"""

    pairs = metadata.depth_order_pairs
    if not pairs:
        return None
    correct = 0
    total = 0
    for front_id, back_id in pairs:
        if front_id >= masks.shape[0] or back_id >= masks.shape[0]:
            continue
        fm, bm = masks[front_id] > 0.5, masks[back_id] > 0.5
        if fm.sum() == 0 or bm.sum() == 0:
            continue
        fd = torch.median(render_depth[fm])
        bd = torch.median(render_depth[bm])
        correct += int(float(fd) <= float(bd))
        total += 1
    return float(correct / total) if total else None


def occlusion_recall_proxy(ownership: torch.Tensor, masks: torch.Tensor, metadata: StressMetadata) -> float | None:
    """被遮挡物体 visible 区域内的 ownership 保持率。"""

    back_ids = sorted({pair[1] for pair in metadata.occlusion_pairs})
    if not back_ids:
        return None
    argmax = _argmax_ownership(ownership)
    vals = []
    for obj_id in back_ids:
        if obj_id < masks.shape[0] and obj_id < ownership.shape[0]:
            vals.append(_purity_for_object(argmax, masks[obj_id], obj_id))
    return float(sum(vals) / len(vals)) if vals else None


def truncation_stability_proxy(ownership: torch.Tensor, masks: torch.Tensor, metadata: StressMetadata) -> float | None:
    """截断物体 mask 内 ownership purity，用于边界截断稳定性诊断。"""

    argmax = _argmax_ownership(ownership)
    vals = []
    for info in metadata.object_infos:
        if info.is_truncated and info.object_id < masks.shape[0] and info.object_id < ownership.shape[0]:
            vals.append(_purity_for_object(argmax, masks[info.object_id], info.object_id))
    return float(sum(vals) / len(vals)) if vals else None


def _jsonable_float(value):
    if value is None:
        return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def compute_stress_metrics(render, masks: torch.Tensor, metadata: StressMetadata) -> dict:
    """计算 lightweight stress diagnostic；这些指标不替代正式 3D metrics。"""

    ownership = render.ownership if hasattr(render, "ownership") else render["ownership"]
    depth = render.depth if hasattr(render, "depth") else render.get("depth", torch.ones_like(masks[0]))
    masks = masks.to(ownership.device).float()
    depth = depth.to(ownership.device).float()
    return {
        "SwapRateProxy": _jsonable_float(swap_rate_proxy(ownership, masks, metadata)),
        "OrderAccProxy": _jsonable_float(order_accuracy_proxy(depth, masks, metadata)),
        "OcclusionRecallProxy": _jsonable_float(occlusion_recall_proxy(ownership, masks, metadata)),
        "TruncationStabilityProxy": _jsonable_float(truncation_stability_proxy(ownership, masks, metadata)),
        "NumOcclusionPairs": int(len(metadata.occlusion_pairs)),
        "NumSameCategoryPairs": int(len(metadata.same_category_pairs)),
        "Subset": metadata.subset,
    }

