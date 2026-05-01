from __future__ import annotations

import torch


def validate_render_output(render, num_objects: int, height: int, width: int, strict: bool = True) -> dict:
    """验证 RenderOutput contract。

    真实 renderer 后续必须满足这个接口；contributions / ownership 是 scene-coupled ownership 的核心输出。
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    def check_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        if tuple(tensor.shape) != shape:
            errors.append(f"{name} shape expected {shape}, got {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all()):
            errors.append(f"{name} contains NaN or Inf")

    check_shape("rgb", render.rgb, (3, height, width))
    check_shape("alpha", render.alpha, (height, width))
    check_shape("depth", render.depth, (height, width))
    check_shape("contributions", render.contributions, (num_objects, height, width))
    check_shape("ownership", render.ownership, (num_objects, height, width))
    check_shape("bg_ownership", render.bg_ownership, (height, width))

    alpha_min = float(render.alpha.detach().min().cpu())
    alpha_max = float(render.alpha.detach().max().cpu())
    stats.update({"alpha_min": alpha_min, "alpha_max": alpha_max})
    if alpha_min < -1e-4 or alpha_max > 1.0001:
        (errors if strict else warnings).append(f"alpha range outside [0,1]: {alpha_min:.4f}, {alpha_max:.4f}")
    if float(render.contributions.detach().min().cpu()) < -1e-6:
        (errors if strict else warnings).append("contributions contain negative values")
    ownership_sum = render.bg_ownership + render.ownership.sum(dim=0)
    diff = (ownership_sum - 1.0).abs()
    stats.update(
        {
            "ownership_sum_min": float(ownership_sum.min().detach().cpu()),
            "ownership_sum_max": float(ownership_sum.max().detach().cpu()),
            "ownership_sum_mean": float(ownership_sum.mean().detach().cpu()),
            "ownership_sum_abs_mean": float(diff.mean().detach().cpu()),
            "contribution_min": float(render.contributions.min().detach().cpu()),
        }
    )
    if stats["ownership_sum_abs_mean"] > 1e-3:
        (errors if strict else warnings).append(f"ownership normalization error mean={stats['ownership_sum_abs_mean']:.6f}")
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings, "stats": stats}


def assert_render_output_contract(render, num_objects: int, height: int, width: int) -> None:
    result = validate_render_output(render, num_objects, height, width, strict=True)
    if not result["valid"]:
        raise AssertionError(result["errors"])
