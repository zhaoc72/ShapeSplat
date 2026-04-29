from __future__ import annotations

from typing import Any, Dict

import torch

from shapesplat.frontend.pipeline import FrontEndOutput
from shapesplat.shape_prior.toy_shape_bank import ToyShapeBank
from shapesplat.shape_prior.retrieval import retrieve_shapes
from shapesplat.shape_prior.soft_support import SoftSupportField
from .object_buffer import GaussianObject
from .scene import ObjectGaussianScene, ObjectMeta


def _opacity_to_logit(p: float, device: torch.device) -> torch.Tensor:
    p = torch.tensor(p, device=device).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def initialize_scene(frontend_output: FrontEndOutput, cfg: Dict[str, Any]) -> ObjectGaussianScene:
    """从 front-end 输出初始化 visible-hidden Gaussian scene。

    hidden branch 是 plausible completion，不是恢复真实隐藏几何；
    它的数量和约束强度由检索置信度控制，低 confidence 时允许 K_hidden=0。
    """
    front = frontend_output
    device = front.image.device
    bank = ToyShapeBank(front.descriptors.shape[1], device)
    ab = cfg.get("ablation", {})
    # DINO retrieval 消融：不使用 instance descriptor 检索，固定使用第一个 toy shape。
    # 这用于测试 mask-guided DINO descriptor retrieval 对 hidden completion 的贡献。
    if ab.get("use_dino_retrieval", True):
        retrieved, weights, conf = retrieve_shapes(front.descriptors, bank, cfg["retrieval"]["top_k"])
    else:
        retrieved = [[bank.shapes[0]] for _ in range(front.descriptors.shape[0])]
        weights = torch.ones((front.descriptors.shape[0], 1), device=device)
        conf = torch.ones((front.descriptors.shape[0],), device=device)
    objects, metas = [], []
    gcfg, rcfg = cfg["gaussians"], cfg["retrieval"]

    for n in range(front.masks.shape[0]):
        mask = front.masks[n]
        ys, xs = torch.where(mask > 0.5)
        area = int(xs.numel())
        k_vis = int(max(gcfg["visible_min"], min(gcfg["visible_max"], area * gcfg["visible_density"])))
        perm = torch.randperm(area, device=device)[:k_vis]
        pix_y, pix_x = ys[perm].float(), xs[perm].float()
        uv = torch.stack([pix_x, pix_y], dim=-1)
        depth = front.depth[ys[perm], xs[perm]]
        vis_means = front.camera.unproject(uv, depth)
        vis_colors = front.image[:, ys[perm], xs[perm]].T.clamp(1e-4, 1 - 1e-4)

        # confidence weighting 消融：关闭后 hidden budget 固定为最大 strength=1。
        if ab.get("use_confidence_weighting", True):
            s = ((conf[n] - rcfg["confidence_low"]) / (rcfg["confidence_high"] - rcfg["confidence_low"])).clamp(0, 1)
            retrieval_confidence = conf[n].detach()
        else:
            s = torch.tensor(1.0, device=device)
            retrieval_confidence = torch.tensor(1.0, device=device)
        # hidden branch 消融：关闭后不创建 hidden Gaussians。
        hidden_enabled = bool(gcfg["use_hidden"] and ab.get("use_hidden_branch", True))
        k_hid = int(torch.floor(s * gcfg["hidden_base"]).item()) if hidden_enabled else 0
        support_field = SoftSupportField(retrieved[n], weights[n], rcfg["support_sigma"]) if retrieved else None

        if k_hid > 0:
            # 从 top-1 shape 采样并粗略对齐到 visible center。真实版本应做尺度/朝向/可见轮廓拟合。
            shape_pts = retrieved[n][0].points.to(device)
            idx = torch.randint(0, shape_pts.shape[0], (k_hid,), device=device)
            center = vis_means.mean(dim=0, keepdim=True)
            scale = vis_means.std(dim=0, keepdim=True).mean().clamp_min(0.05) * 1.6
            hid_means = center + shape_pts[idx] * scale
            hid_colors = vis_colors.mean(dim=0, keepdim=True).expand(k_hid, 3)
        else:
            hid_means = torch.zeros((0, 3), device=device)
            hid_colors = torch.zeros((0, 3), device=device)

        means = torch.cat([vis_means, hid_means], dim=0)
        colors = torch.cat([vis_colors, hid_colors], dim=0).clamp(1e-4, 1 - 1e-4)
        colors_logits = torch.log(colors / (1 - colors))
        log_scales = torch.full_like(means, float(gcfg["init_log_scale"]))
        vis_op = _opacity_to_logit(gcfg["init_opacity"], device).expand(k_vis, 1)
        hid_op = _opacity_to_logit(gcfg["hidden_opacity"], device).expand(k_hid, 1)
        opacity_logits = torch.cat([vis_op, hid_op], dim=0)
        branch_ids = torch.cat([torch.zeros(k_vis, device=device), torch.ones(k_hid, device=device)]).long()
        # visible-hidden split 消融：即使存在 hidden samples，也把它们视为 visible branch，
        # 模拟没有 factorized representation 的单一 object Gaussian buffer。
        if not ab.get("use_visible_hidden_split", True):
            branch_ids = torch.zeros_like(branch_ids)
        obj = GaussianObject(means, log_scales, opacity_logits, colors_logits, branch_ids, n)
        objects.append(obj)
        metas.append(ObjectMeta(n, mask.detach(), front.descriptors[n].detach(), retrieval_confidence, support_field, k_vis, k_hid))

    if not objects:
        raise RuntimeError("SAM3 stub 没有产生任何前景 mask，无法初始化 scene。")
    return ObjectGaussianScene(objects, metas)
