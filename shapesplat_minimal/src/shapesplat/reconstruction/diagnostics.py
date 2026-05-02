from __future__ import annotations

import torch

from shapesplat.renderer.contract import validate_render_output


def _float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _object_counts(scene) -> tuple[list[int], list[int], list[int]]:
    total, visible, hidden = [], [], []
    for obj in getattr(scene, "objects", []):
        branch = getattr(obj, "branch_ids", None)
        if branch is None:
            k = int(obj.means.shape[0])
            total.append(k)
            visible.append(k)
            hidden.append(0)
        else:
            total.append(int(branch.numel()))
            visible.append(int((branch == 0).sum().detach().cpu()))
            hidden.append(int((branch == 1).sum().detach().cpu()))
    return total, visible, hidden


def compute_reconstruction_diagnostics(scene, render, front, cfg) -> dict:
    """计算主方法诊断信息。

    这些 diagnostics 用于定位 Ours 在正式 benchmark 上的失败模式，例如 ownership
    未归一化、hidden branch 没有初始化、shape prior 退回 toy bank 或 renderer fallback。
    它们不是论文核心指标。
    """
    warnings: list[str] = []
    total, visible, hidden = _object_counts(scene)
    ownership_sum = render.bg_ownership + render.ownership.sum(dim=0)
    fg = front.masks.sum(dim=0).clamp(0, 1) > 0.5
    bg = ~fg
    contract = validate_render_output(render, int(front.masks.shape[0]), int(front.image.shape[-2]), int(front.image.shape[-1]), strict=False)

    retrieval_conf = []
    for meta in getattr(scene, "metas", []):
        retrieval_conf.append(_float(getattr(meta, "retrieval_confidence", torch.tensor(0.0))))

    renderer_extras = getattr(render, "extras", {}) or {}
    shape_backend = cfg.get("shape_bank", {}).get("backend", "unknown")
    renderer_backend = renderer_extras.get("renderer_backend", cfg.get("renderer", {}).get("backend", "unknown"))
    frontend_cfg = cfg.get("frontend", {})
    cache_cfg = cfg.get("frontend_cache", {})

    if shape_backend in {"toy", "auto"} and cfg.get("shape_bank", {}).get("root") in (None, ""):
        warnings.append("Using ToyShapeBank or auto fallback; this is not suitable for final submission.")
    if renderer_backend == "soft" or cfg.get("renderer", {}).get("backend") in {"soft", "auto"}:
        warnings.append("Using SoftGaussianRenderer or possible soft fallback; report this clearly for final experiments.")
    if frontend_cfg.get("sam_backend") == "stub":
        warnings.append("Using Sam3Stub masks; final experiments should use fixed real masks or benchmark masks.")
    if frontend_cfg.get("dino_backend") == "stub":
        warnings.append("Using DinoV3Stub descriptors; shape retrieval diagnostics are debug-only.")
    if frontend_cfg.get("depth_backend") == "stub":
        warnings.append("Using DepthStub; depth diagnostics are debug-only.")
    if sum(hidden) == 0:
        warnings.append("Hidden branch has zero Gaussians for all objects.")

    depth = front.depth.detach()
    return {
        "object_counts": {
            "num_objects": int(front.masks.shape[0]),
            "gaussian_count_total": int(sum(total)),
            "gaussian_count_per_object": total,
            "visible_gaussian_count_per_object": visible,
            "hidden_gaussian_count_per_object": hidden,
        },
        "ownership": {
            "ownership_sum_mean": _float(ownership_sum.mean()),
            "ownership_sum_min": _float(ownership_sum.min()),
            "ownership_sum_max": _float(ownership_sum.max()),
            "bg_ownership_mean": _float(render.bg_ownership.mean()),
            "max_ownership_mean": _float(render.ownership.max(dim=0).values.mean()),
        },
        "leakage": {
            "alpha_bg_mean": _float(render.alpha[bg].mean()) if bool(bg.any()) else 0.0,
            "alpha_fg_mean": _float(render.alpha[fg].mean()) if bool(fg.any()) else 0.0,
        },
        "retrieval_prior": {
            "retrieval_confidence_per_object": retrieval_conf,
            "hidden_prior_enabled": bool(cfg.get("ablation", {}).get("use_hidden_prior", True)),
            "shape_bank_backend": shape_backend,
        },
        "renderer": {
            "renderer_backend": renderer_backend,
            "renderer_fallback": renderer_extras.get("fallback", None),
            "render_contract_valid": bool(contract["valid"]),
            "render_contract_warnings": contract["warnings"],
            "render_contract_errors": contract["errors"],
        },
        "frontend": {
            "mask_source": frontend_cfg.get("mask_source", "unknown"),
            "frontend_cache_used": bool(cache_cfg.get("use_cache", False)),
            "descriptor_dim": int(front.descriptors.shape[1]) if front.descriptors.ndim == 2 else None,
            "depth_min": _float(depth.min()),
            "depth_max": _float(depth.max()),
            "depth_mean": _float(depth.mean()),
        },
        "warnings": warnings,
    }
