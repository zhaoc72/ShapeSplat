from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .toy_shape_bank import ToyShapeBank, ToyShape


def retrieve_shapes(descriptors: torch.Tensor, shape_bank: ToyShapeBank, top_k: int) -> Tuple[List[List[ToyShape]], torch.Tensor, torch.Tensor]:
    """用 cosine similarity 从 shape bank 检索 top-K。

    retrieval confidence 后续控制 hidden Gaussian budget 和 hidden prior loss strength：
    置信度低时少生成甚至不生成 hidden branch，避免模板幻觉污染可见重建。
    confidence = sigmoid(2 * top1 + 2 * margin)
    """
    if descriptors.shape[0] == 0:
        return [], torch.zeros((0, top_k), device=descriptors.device), torch.zeros((0,), device=descriptors.device)
    sims = descriptors @ shape_bank.descriptors.T
    k = min(top_k, sims.shape[1])
    vals, idx = torch.topk(sims, k=k, dim=1)
    weights = F.softmax(vals, dim=1)
    if k > 1:
        margin = vals[:, 0] - vals[:, 1]
    else:
        margin = vals[:, 0]
    confidence = torch.sigmoid(2 * vals[:, 0] + 2 * margin)
    retrieved = [[shape_bank.shapes[int(j)] for j in row] for row in idx]
    return retrieved, weights, confidence
