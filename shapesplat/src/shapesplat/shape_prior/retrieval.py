from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .types import ShapeAsset


def _asset_similarity(
    descriptors: torch.Tensor,
    asset: ShapeAsset,
    use_multi_view_descriptors: bool,
) -> torch.Tensor:
    """计算一批 object descriptor 与单个 shape asset 的 cosine similarity。

    真实 shape bank 可能保存一个全局 descriptor，也可能保存多视角 descriptor。
    多视角时取 max-over-views，可以弱化单图视角与 shape 预渲染视角不一致的问题。
    """
    if use_multi_view_descriptors and asset.descriptors is not None:
        bank_desc = asset.descriptors.to(descriptors.device, descriptors.dtype)
        if bank_desc.ndim != 2:
            raise ValueError(f"Shape {asset.shape_id} descriptors must be [V,D], got {tuple(bank_desc.shape)}")
        if bank_desc.shape[1] != descriptors.shape[1]:
            raise ValueError(
                f"Descriptor dimension mismatch for shape {asset.shape_id}: "
                f"query D={descriptors.shape[1]}, bank D={bank_desc.shape[1]}"
            )
        bank_desc = F.normalize(bank_desc, dim=1)
        query = F.normalize(descriptors, dim=1)
        return (query @ bank_desc.T).amax(dim=1)

    if asset.descriptor is None:
        raise ValueError(f"Shape {asset.shape_id} has no descriptor. Provide descriptor or descriptors.")
    bank_desc = asset.descriptor.to(descriptors.device, descriptors.dtype)
    if bank_desc.ndim != 1:
        raise ValueError(f"Shape {asset.shape_id} descriptor must be [D], got {tuple(bank_desc.shape)}")
    if bank_desc.shape[0] != descriptors.shape[1]:
        raise ValueError(
            f"Descriptor dimension mismatch for shape {asset.shape_id}: "
            f"query D={descriptors.shape[1]}, bank D={bank_desc.shape[0]}"
        )
    return F.normalize(descriptors, dim=1) @ F.normalize(bank_desc, dim=0)


def retrieve_shapes(
    descriptors: torch.Tensor,
    shape_bank,
    top_k: int,
    use_multi_view_descriptors: bool = True,
    temperature: float = 0.07,
) -> Tuple[List[List[ShapeAsset]], torch.Tensor, torch.Tensor]:
    """根据 object descriptor 从 shape bank 检索 top-K shape。

    返回的 confidence 会继续控制 hidden Gaussian budget 和 hidden support prior
    strength。这里的检索只是 weak prior：它给 hidden branch 一个可能的形状支撑，
    不是 hard template fitting，也不代表真实隐藏几何已经被恢复。
    """
    assets: list[ShapeAsset] = list(getattr(shape_bank, "assets", []))
    device = descriptors.device
    if descriptors.shape[0] == 0:
        return [], torch.zeros((0, 0), device=device), torch.zeros((0,), device=device)
    if not assets:
        raise ValueError("Shape bank is empty; cannot retrieve shapes.")

    sims = torch.stack(
        [_asset_similarity(descriptors, asset, use_multi_view_descriptors) for asset in assets],
        dim=1,
    )
    k = min(max(int(top_k), 1), sims.shape[1])
    vals, idx = torch.topk(sims, k=k, dim=1)
    temperature = max(float(temperature), 1e-6)
    weights = F.softmax(vals / temperature, dim=1)

    if k > 1:
        margin = vals[:, 0] - vals[:, 1]
    else:
        # 只有一个 shape 时没有真实 margin，用固定正 margin 避免 confidence 永远偏低。
        margin = torch.full_like(vals[:, 0], 0.5)
    confidence = torch.sigmoid(2.0 * vals[:, 0] + 2.0 * margin)
    retrieved = [[assets[int(j)] for j in row] for row in idx]
    return retrieved, weights, confidence
