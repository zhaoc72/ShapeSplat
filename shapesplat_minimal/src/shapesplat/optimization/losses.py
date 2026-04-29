from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from shapesplat.frontend.pipeline import FrontEndOutput
from shapesplat.gaussian.scene import ObjectGaussianScene
from shapesplat.geometry.masks import union_mask
from shapesplat.renderer.soft_renderer import RenderOutput, SoftGaussianRenderer


def _zero(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def compute_losses(
    scene: ObjectGaussianScene,
    renderer: SoftGaussianRenderer,
    render: RenderOutput,
    front: FrontEndOutput,
    cfg: Dict[str, Any],
    stage: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算分阶段损失。

    visible_rgb/scene 防 object drift；visible_alpha/identity 防 object merging；
    hidden_prior/bridge 防 hidden hallucination；bg 防前景泄漏；edit 防编辑 collateral change。
    """
    w = cfg["loss_weights"]
    ab = cfg.get("ablation", {})
    device = front.image.device
    fg = union_mask(front.masks).to(device)
    bg = 1.0 - fg
    eps = 1e-6
    losses: Dict[str, torch.Tensor] = {}

    if fg.sum() > 0:
        rgb_l1 = ((render.rgb - front.image).abs() * fg[None]).sum() / (fg.sum() * 3).clamp_min(1)
        losses["visible_rgb"] = rgb_l1
        losses["scene"] = rgb_l1
        losses["visible_depth"] = ((render.depth - front.depth).abs() * fg).sum() / fg.sum().clamp_min(1)
    else:
        losses["visible_rgb"] = losses["scene"] = losses["visible_depth"] = _zero(device)

    # per-object ownership BCE：显式压住 attribution，避免多个物体混成一个 blob。
    own = render.ownership.clamp(eps, 1 - eps)
    if front.masks.shape[0] > 0:
        losses["visible_alpha"] = F.binary_cross_entropy(own, front.masks.float(), reduction="mean")
        label = front.masks.argmax(dim=0)
        reliable = fg > 0.5
        if reliable.any():
            logits = torch.log(own.clamp_min(eps))
            losses["identity"] = F.nll_loss(logits.permute(1, 2, 0)[reliable], label[reliable])
        else:
            losses["identity"] = _zero(device)
    else:
        losses["visible_alpha"] = losses["identity"] = _zero(device)

    losses["bg"] = (render.alpha.square() * bg).sum() / bg.sum().clamp_min(1)

    prior_terms, bridge_terms = [], []
    for obj, meta in zip(scene.objects, scene.metas):
        hid = obj.hidden_mask
        vis = obj.visible_mask
        if bool(hid.any()) and meta.support_field is not None:
            v = obj.means[vis]
            h = obj.means[hid]
            center = v.mean(dim=0, keepdim=True).detach()
            scale = v.std(dim=0, keepdim=True).mean().detach().clamp_min(0.05)
            local_h = (h - center) / scale
            phi = meta.support_field.support(local_h)
            prior_terms.append((-torch.log(phi.clamp_min(1e-5))).mean() * meta.retrieval_confidence.to(device))
            bridge_terms.append(torch.cdist(h, v).amin(dim=1).mean())
    losses["hidden_prior"] = torch.stack(prior_terms).mean() if prior_terms else _zero(device)
    losses["bridge"] = torch.stack(bridge_terms).mean() if bridge_terms else _zero(device)

    regs = []
    for obj in scene.objects:
        regs.append(obj.log_scales.square().mean() + obj.opacity_logits.sigmoid().mean() * 0.01)
    losses["reg"] = torch.stack(regs).mean()
    # layout loss 当前 minimal 版本只是预留项；保留 key 方便 ablation 日志对齐。
    losses["layout"] = _zero(device)

    # Ablation gates：保留 term key，但把被关闭模块置为 device 上的 0 tensor。
    # scene loss: 跨物体组合一致性；关闭后只保留 visible_rgb 等局部约束。
    if not ab.get("use_scene_loss", True):
        losses["scene"] = _zero(device)
    # identity loss: object ownership / same-category swap；关闭后更容易发生物体归属混淆。
    if not ab.get("use_ownership_loss", True):
        losses["identity"] = _zero(device)
    # weak depth loss: 单图 layout cue；关闭后深度排序约束变弱。
    if not ab.get("use_depth_loss", True):
        losses["visible_depth"] = _zero(device)
    # hidden prior: hidden completion；关闭后 hidden branch 不受 soft shape support 约束。
    if not ab.get("use_hidden_prior", True):
        losses["hidden_prior"] = _zero(device)
    # bridge: visible-hidden continuity；关闭后 hidden 与 visible 更容易断裂。
    if not ab.get("use_bridge_loss", True):
        losses["bridge"] = _zero(device)
    # bg loss: foreground/background leakage；关闭后 alpha 更可能泄漏到背景。
    if not ab.get("use_bg_loss", True):
        losses["bg"] = _zero(device)
    if not ab.get("use_layout_loss", True):
        losses["layout"] = _zero(device)

    active = {
        "visible": ["visible_rgb", "visible_alpha", "visible_depth", "scene", "identity", "bg", "layout", "reg"],
        "hidden": ["visible_rgb", "visible_alpha", "hidden_prior", "bridge", "bg", "reg"],
        "joint": ["visible_rgb", "visible_alpha", "visible_depth", "scene", "identity", "hidden_prior", "bridge", "bg", "layout", "reg"],
        "edit": ["visible_rgb", "visible_alpha", "scene", "identity", "hidden_prior", "bridge", "bg", "layout", "reg"],
    }[stage]
    total = sum(w[name] * losses[name] for name in active)
    terms = {k: float(v.detach().cpu()) for k, v in losses.items()}
    terms["total"] = float(total.detach().cpu())
    return total, terms
