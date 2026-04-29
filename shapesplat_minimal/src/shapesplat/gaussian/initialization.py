from __future__ import annotations

from typing import Any, Dict

import torch

from shapesplat.frontend.pipeline import FrontEndOutput
from shapesplat.shape_prior.retrieval import retrieve_shapes
from shapesplat.shape_prior.shape_bank_backend import build_shape_bank
from shapesplat.shape_prior.soft_support import SoftSupportField
from .object_buffer import GaussianObject
from .scene import ObjectGaussianScene, ObjectMeta


def _opacity_to_logit(p: float, device: torch.device) -> torch.Tensor:
    p_tensor = torch.tensor(p, device=device).clamp(1e-4, 1 - 1e-4)
    return torch.log(p_tensor / (1 - p_tensor))


def initialize_scene(frontend_output: FrontEndOutput, cfg: Dict[str, Any]) -> ObjectGaussianScene:
    """根据 front-end 输出初始化 visible-hidden Gaussian scene。

    visible branch 由 retained visible mask 和 weak depth 直接初始化，承担可见区域强监督；
    hidden branch 来自 shape retrieval 的 soft prior，只表示 plausible completion，不是 GT
    隐藏几何。v0.8 以后 shape bank 可以是 toy，也可以是本地点云 file bank。
    """
    front = frontend_output
    device = front.image.device
    ab = cfg.get("ablation", {})
    gcfg, rcfg = cfg["gaussians"], cfg["retrieval"]

    descriptor_dim = int(front.descriptors.shape[1])
    shape_bank = build_shape_bank(cfg, descriptor_dim=descriptor_dim, device=device)
    assets = list(getattr(shape_bank, "assets", []))

    # dino_backend 与 use_dino_retrieval 是两层不同开关：
    # dino_backend 控制 descriptor extractor 是 stub 还是真实 DINOv3；
    # use_dino_retrieval 控制 shape retrieval 是否使用这些 descriptor。
    if ab.get("use_dino_retrieval", True):
        retrieved, weights, conf = retrieve_shapes(
            front.descriptors,
            shape_bank,
            top_k=rcfg["top_k"],
            use_multi_view_descriptors=rcfg.get("use_multi_view_descriptors", True),
            temperature=rcfg.get("temperature", 0.07),
        )
    else:
        if not assets:
            raise RuntimeError("Shape bank is empty; no-DINO retrieval ablation cannot select a fallback shape.")
        retrieved = [[assets[0]] for _ in range(front.descriptors.shape[0])]
        weights = torch.ones((front.descriptors.shape[0], 1), device=device)
        conf = torch.ones((front.descriptors.shape[0],), device=device)

    objects, metas = [], []
    for n in range(front.masks.shape[0]):
        mask = front.masks[n]
        ys, xs = torch.where(mask > 0.5)
        area = int(xs.numel())
        if area == 0:
            continue

        k_vis = int(max(gcfg["visible_min"], min(gcfg["visible_max"], area * gcfg["visible_density"])))
        k_vis = min(k_vis, area)
        perm = torch.randperm(area, device=device)[:k_vis]
        pix_y, pix_x = ys[perm].float(), xs[perm].float()
        uv = torch.stack([pix_x, pix_y], dim=-1)

        # depth 是 weak initialization/layout cue，不是 GT geometry。真实 depth wrapper
        # 输出也会先归一化到 canonical z range，这里仍做有限性保护。
        sampled_depth = front.depth[ys[perm], xs[perm]]
        fg_depth = front.depth[mask > 0.5]
        valid_fg = torch.isfinite(fg_depth) & (fg_depth > 0)
        fill_depth = (
            fg_depth[valid_fg].median()
            if bool(valid_fg.any())
            else torch.tensor(cfg["camera"]["z_near"], device=device)
        )
        depth = torch.where(torch.isfinite(sampled_depth) & (sampled_depth > 0), sampled_depth, fill_depth)
        vis_means = front.camera.unproject(uv, depth)
        vis_colors = front.image[:, ys[perm], xs[perm]].T.clamp(1e-4, 1 - 1e-4)

        # confidence weighting 控制 hidden Gaussian budget 和 hidden prior strength。
        # 消融关闭时固定 strength=1，用于观察 retrieval confidence 的贡献。
        if ab.get("use_confidence_weighting", True):
            s = ((conf[n] - rcfg["confidence_low"]) / (rcfg["confidence_high"] - rcfg["confidence_low"])).clamp(0, 1)
            retrieval_confidence = conf[n].detach()
        else:
            s = torch.tensor(1.0, device=device)
            retrieval_confidence = torch.tensor(1.0, device=device)

        hidden_enabled = bool(gcfg["use_hidden"] and ab.get("use_hidden_branch", True) and len(retrieved[n]) > 0)
        k_hid = int(torch.floor(s * gcfg["hidden_base"]).item()) if hidden_enabled else 0
        support_field = SoftSupportField(retrieved[n], weights[n], rcfg["support_sigma"]) if hidden_enabled else None

        if k_hid > 0:
            # 从检索 shape 的 top-1 point cloud 采样 hidden centers，并粗略对齐到 visible
            # Gaussian 的中心和尺度附近。这里不做 hard fitting，只给 hidden branch 一个可优化起点。
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
        branch_ids = torch.cat(
            [torch.zeros(k_vis, device=device), torch.ones(k_hid, device=device)],
            dim=0,
        ).long()

        # visible-hidden split 消融：hidden samples 仍可存在，但全部视作 visible branch，
        # 用来模拟没有 factorized representation 的单一 object Gaussian buffer。
        if not ab.get("use_visible_hidden_split", True):
            branch_ids = torch.zeros_like(branch_ids)

        obj = GaussianObject(means, log_scales, opacity_logits, colors_logits, branch_ids, n)
        objects.append(obj)
        metas.append(
            ObjectMeta(
                n,
                mask.detach(),
                front.descriptors[n].detach(),
                retrieval_confidence,
                support_field,
                k_vis,
                k_hid,
            )
        )

    if not objects:
        raise RuntimeError("No visible masks are available, so Gaussian scene initialization cannot proceed.")
    return ObjectGaussianScene(objects, metas)
