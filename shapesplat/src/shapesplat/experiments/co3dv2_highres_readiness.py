from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shapesplat.cache.validate_cache import validate_frontend_cache_manifest
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.manifest import load_manifest
from shapesplat.frontend.dinov3_dependency_check import check_dinov3_dependencies
from shapesplat.runtime.cuda_check import check_cuda_runtime


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _add_check(checks: list[dict], name: str, ok: bool, message: str, level: str = "info") -> None:
    checks.append({"name": name, "ok": bool(ok), "level": level, "message": message})


def check_co3dv2_highres_ready(
    config: str | Path | dict,
    manifest: str | Path,
    cache_manifest: str | Path | None = None,
    strict: bool = False,
) -> dict:
    """检查 CO3Dv2 high-res diagnostic 是否准备好。

    中文注释：CO3Dv2 single 是 real-image diagnostic / single foreground benchmark，
    readiness 会明确指出 SoftRenderer、ToyShapeBank、缺 cache 等情况是否只适合诊断，
    避免把 debug 结果误写成论文最终结果。
    """
    manifest_path = Path(manifest)
    warnings: list[str] = []
    errors: list[str] = []
    checks: list[dict] = []
    summary: dict[str, Any] = {}
    if isinstance(config, dict):
        cfg = config
    else:
        try:
            cfg = load_config(config)
        except Exception as exc:
            # 中文注释：readiness 本身应给出诊断报告，而不是在 CUDA/配置不可用时直接崩掉。
            errors.append(f"failed to load config with requested runtime: {exc}")
            cfg = load_config(config, runtime_overrides={"device": "cpu", "allow_cpu_fallback": True, "require_cuda_for_experiments": False})

    manifest_ok = manifest_path.exists() and manifest_path.is_file()
    _add_check(checks, "manifest_exists", manifest_ok, str(manifest_path), "error" if not manifest_ok else "info")
    if not manifest_ok:
        errors.append(f"benchmark manifest missing: {manifest_path}")
        records = []
    else:
        try:
            records = load_manifest(manifest_path)
            _add_check(checks, "manifest_loadable", len(records) > 0, f"num_records={len(records)}")
            summary["num_manifest_records"] = len(records)
        except Exception as exc:
            records = []
            errors.append(f"failed to load manifest: {exc}")
            _add_check(checks, "manifest_loadable", False, str(exc), "error")

    if records:
        first = records[0]
        try:
            image = load_image(first.image_path, resize_mode="none")
            image_hw = list(image.shape[-2:])
            summary["sample_original_image_shape"] = list(image.shape)
            highish = max(image_hw) >= 448
            _add_check(checks, "sample_image_highres", highish, f"sample_hw={image_hw}", "warning" if not highish else "info")
            if not highish:
                warnings.append("sample image appears below high-res diagnostic target")
        except Exception as exc:
            warnings.append(f"could not inspect sample image shape: {exc}")
            _add_check(checks, "sample_image_highres", False, str(exc), "warning")
        mask_path = first.metadata.get("mask_path")
        _add_check(checks, "file_mask_present", bool(mask_path and Path(mask_path).exists()), str(mask_path), "error" if not mask_path else "info")
        if not mask_path or not Path(mask_path).exists():
            errors.append("CO3Dv2 high-res workflow requires file masks in manifest.")

    image_cfg = cfg.get("image", {})
    front_cfg = cfg.get("frontend", {})
    renderer_cfg = cfg.get("renderer", {})
    shape_cfg = cfg.get("shape_bank", {})
    debug_cfg = cfg.get("debug", {})

    long_side = int(image_cfg.get("long_side") or image_cfg.get("size") or 0)
    dino_input = int(front_cfg.get("dino_input_size") or 0)
    resize_ok = image_cfg.get("resize_mode") == "keep_aspect" and long_side >= 448
    dino_ok = dino_input >= 448
    nearest_ok = front_cfg.get("mask_resize_mode") == "nearest"
    cap_disabled = debug_cfg.get("allow_debug_iteration_cap") is False
    _add_check(checks, "highres_image_config", resize_ok, f"resize_mode={image_cfg.get('resize_mode')} long_side={long_side}", "error" if strict and not resize_ok else "warning")
    _add_check(checks, "dino_input_highres", dino_ok, f"dino_input_size={dino_input}", "error" if strict and not dino_ok else "warning")
    _add_check(checks, "mask_resize_nearest", nearest_ok, f"mask_resize_mode={front_cfg.get('mask_resize_mode')}", "error" if not nearest_ok else "info")
    _add_check(checks, "debug_cap_disabled", cap_disabled, f"allow_debug_iteration_cap={debug_cfg.get('allow_debug_iteration_cap')}", "error" if strict and not cap_disabled else "warning")
    if not resize_ok:
        (errors if strict else warnings).append("high-res image config should use keep_aspect and long_side >= 448.")
    if not dino_ok:
        (errors if strict else warnings).append("DINOv3 input size should be at least 448 for CO3Dv2 high-res diagnostics.")
    if not nearest_ok:
        errors.append("file mask resize must use nearest.")
    if not cap_disabled:
        (errors if strict else warnings).append("debug iteration cap should be disabled for high-res CO3Dv2 diagnostics.")

    checkpoint = front_cfg.get("dino_checkpoint")
    checkpoint_ok = bool(checkpoint and Path(str(checkpoint)).exists())
    _add_check(checks, "dinov3_checkpoint_exists", checkpoint_ok, str(checkpoint), "error" if not checkpoint_ok else "info")
    if not checkpoint_ok:
        errors.append(f"DINOv3 checkpoint missing: {checkpoint}")

    deps = check_dinov3_dependencies()
    _add_check(checks, "dinov3_dependencies", bool(deps.get("ok")), json.dumps(deps.get("missing_required", [])), "error" if not deps.get("ok") else "info")
    if not deps.get("ok"):
        errors.append(f"DINOv3 dependencies missing: {deps.get('missing_required')}. Suggested: {deps.get('install_command')}")

    cuda = check_cuda_runtime(cfg)
    cuda_ok = cuda.get("status") == "cuda_ok"
    _add_check(checks, "cuda_available", cuda_ok, cuda.get("status", "unknown"), "warning" if not cuda_ok else "info")
    if not cuda_ok:
        warnings.append("CUDA is not confirmed usable in readiness check; real high-res DINOv3 cache should run on CUDA.")

    cache_path = Path(cache_manifest) if cache_manifest else Path(str(cfg.get("frontend_cache", {}).get("cache_manifest") or ""))
    if cache_path and str(cache_path) not in ("", "."):
        cache_exists = cache_path.exists()
        _add_check(checks, "cache_manifest_exists", cache_exists, str(cache_path), "error" if strict and not cache_exists else "warning")
        if cache_exists:
            cache_report = validate_frontend_cache_manifest(cache_path)
            summary["cache_validation"] = cache_report
            cache_valid = bool(cache_report.get("valid")) and int(cache_report.get("num_valid", 0)) > 0
            _add_check(checks, "cache_manifest_valid", cache_valid, f"num_valid={cache_report.get('num_valid')}", "error" if strict and not cache_valid else "warning")
            if not cache_valid:
                (errors if strict else warnings).append("frontend cache manifest is missing valid high-res entries.")
        elif strict:
            errors.append(f"cache manifest missing: {cache_path}")

    soft = renderer_cfg.get("backend") == "soft" or renderer_cfg.get("fallback_to_soft", True)
    toy = shape_cfg.get("backend") in ("toy", "auto") and not shape_cfg.get("root")
    _add_check(checks, "renderer_not_final", not soft, f"renderer={renderer_cfg.get('backend')} fallback_to_soft={renderer_cfg.get('fallback_to_soft')}", "error" if strict and soft else "warning")
    _add_check(checks, "shape_bank_not_toy", not toy, f"shape_bank={shape_cfg.get('backend')} root={shape_cfg.get('root')}", "error" if strict and toy else "warning")
    if soft:
        (errors if strict else warnings).append("SoftGaussianRenderer/fallback is diagnostic-only, not paper-final 3DGS quality.")
    if toy:
        (errors if strict else warnings).append("ToyShapeBank/auto fallback is diagnostic-only, not paper-final shape prior.")

    ready = len(errors) == 0
    return {
        "ready": ready,
        "strict": bool(strict),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "summary": _jsonable(summary),
    }


def save_co3dv2_highres_readiness_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "readiness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# CO3Dv2 High-Resolution Readiness",
        "",
        f"- Ready: `{report.get('ready')}`",
        f"- Strict: `{report.get('strict')}`",
        "",
        "## Errors",
        *[f"- {e}" for e in report.get("errors", [])],
        "",
        "## Warnings",
        *[f"- {w}" for w in report.get("warnings", [])],
        "",
        "## Checks",
        "| Check | OK | Level | Message |",
        "|---|---:|---|---|",
    ]
    for check in report.get("checks", []):
        msg = str(check.get("message", "")).replace("|", "\\|")
        lines.append(f"| {check.get('name')} | {check.get('ok')} | {check.get('level')} | {msg} |")
    (out / "readiness.md").write_text("\n".join(lines), encoding="utf-8")
