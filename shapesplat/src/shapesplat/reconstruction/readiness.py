from __future__ import annotations

from pathlib import Path


def _add_check(checks: list[dict], name: str, ok: bool, message: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "message": message})


def check_ours_core_ready(cfg: dict, manifest_path: str | Path | None = None, strict: bool = False) -> dict:
    """检查 Ours 主方法是否适合跑正式 benchmark。

    debug 模式允许 stub/toy/soft fallback；strict=True 用于投稿前检查，此时仍使用
    Sam3Stub、DinoV3Stub、ToyShapeBank 或 SoftGaussianRenderer 会被提升为 error。
    """
    checks: list[dict] = []
    warnings: list[str] = []
    errors: list[str] = []

    if manifest_path is not None:
        exists = Path(manifest_path).exists()
        _add_check(checks, "benchmark_manifest", exists, str(manifest_path))
        if not exists:
            errors.append(f"Benchmark manifest not found: {manifest_path}")

    frontend = cfg.get("frontend", {})
    cache = cfg.get("frontend_cache", {})
    shape = cfg.get("shape_bank", {})
    renderer = cfg.get("renderer", {})
    training = cfg.get("training", {})

    mask_source = frontend.get("mask_source")
    _add_check(checks, "same_mask_file_protocol", mask_source == "file", f"frontend.mask_source={mask_source}")
    if mask_source != "file":
        warnings.append("Final same-mask benchmark should use frontend.mask_source=file or fixed frontend cache masks.")

    _add_check(checks, "frontend_cache_configured", bool(cache.get("cache_manifest") or cache.get("cache_root") or not cache.get("use_cache")), str(cache))
    if cache.get("use_cache") and not (cache.get("cache_manifest") or cache.get("cache_root")):
        warnings.append("frontend_cache.use_cache=true but no cache_manifest/cache_root is configured.")

    _add_check(checks, "shape_bank_configured", bool(shape.get("root") or shape.get("fallback_to_toy", False)), str(shape))
    if shape.get("backend") in {"toy", "auto"} and not shape.get("root"):
        msg = "Shape bank may fallback to ToyShapeBank; debug only."
        (errors if strict else warnings).append(msg)

    if renderer.get("backend") in {"soft", "auto"} and renderer.get("fallback_to_soft", True):
        msg = "Renderer is soft/auto fallback; not a final CUDA 3DGS result unless explicitly reported."
        (errors if strict else warnings).append(msg)
    _add_check(checks, "renderer_backend_configured", bool(renderer.get("backend")), f"renderer.backend={renderer.get('backend')}")

    for name in ("sam_backend", "dino_backend", "depth_backend"):
        value = frontend.get(name)
        if value == "stub":
            msg = f"{name}=stub; suitable for smoke tests only."
            (errors if strict else warnings).append(msg)

    total_iters = sum(int(training.get(k, 0)) for k in ("visible_warmup_iters", "hidden_prior_iters", "joint_ownership_iters", "edit_finetune_iters"))
    _add_check(checks, "training_iterations", total_iters > 0, f"total_iters={total_iters}")
    if total_iters < 20:
        warnings.append("Training iterations are very small; metrics are smoke-test only.")

    ours = cfg.get("ours", {})
    _add_check(checks, "output_saving", bool(ours.get("save_ownership_npy", True) and ours.get("save_diagnostics", True)), str(ours))

    ready = len(errors) == 0
    return {"ready": ready, "warnings": warnings, "errors": errors, "checks": checks}
