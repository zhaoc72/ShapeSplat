from __future__ import annotations

from pathlib import Path

from shapesplat.baselines.method_catalog import load_method_catalog
from shapesplat.cache.validate_cache import validate_frontend_cache_manifest
from shapesplat.config import load_config
from shapesplat.datasets.benchmark.validator_v2 import validate_benchmark_v2
from shapesplat.integration.capabilities import detect_renderer_capability
from shapesplat.reporting.io import save_json


def _check(checks: list[dict], name: str, ok: bool, message: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "message": str(message)})


def check_final_paper_ready(profile: dict, out_dir: str | Path | None = None, strict: bool = False) -> dict:
    """最终论文实验 readiness check。

    strict=False 时 stub/toy/soft 只作为 warning，保证 final_debug 不会因为本地没有真实组件崩溃；
    strict=True 用于投稿前检查，可把 fallback 组件提升为 error。
    """
    checks: list[dict] = []
    warnings: list[str] = []
    errors: list[str] = []
    summary: dict = {}

    bench = profile.get("benchmark", {})
    manifest = bench.get("manifest")
    manifest_ok = bool(manifest and Path(manifest).exists())
    _check(checks, "benchmark_manifest", manifest_ok, manifest or "")
    if manifest_ok:
        report = validate_benchmark_v2(manifest, max_rows=bench.get("max_images"), check_optional_gt=False, check_cache=False)
        summary["benchmark_num_rows"] = report.get("num_rows")
        summary["benchmark_num_valid"] = report.get("num_valid")
        _check(checks, "benchmark_valid", bool(report.get("valid")), f"valid={report.get('valid')}")
        if not report.get("valid"):
            errors.append("Benchmark v2 validation failed.")
        if profile.get("readiness", {}).get("warn_if_missing_geometry_gt", True):
            gt_count = 0
            for row in report.get("rows", []):
                # validator rows 不展开 GT 计数，这里只给整体 warning，避免把 optional geometry 当硬要求。
                pass
            warnings.append("Geometry GT is optional; metrics will be unavailable for rows without gt_pointcloud_path.")
    else:
        # 中文注释：final_debug/profile 可以声明缺失 benchmark 时由 final runner 自动创建；
        # 这种情况不应阻塞 smoke test，但仍要提醒正式投稿前固定 manifest。
        if bench.get("create_if_missing"):
            warnings.append(f"Benchmark manifest missing but profile can create it: {manifest}")
            summary["benchmark_will_create"] = True
        else:
            errors.append(f"Benchmark manifest missing: {manifest}")

    ours_cfg_path = profile.get("ours", {}).get("config", "configs/final_ours.yaml")
    cfg_ok = Path(ours_cfg_path).exists()
    _check(checks, "ours_config", cfg_ok, ours_cfg_path)
    cfg = load_config(ours_cfg_path) if cfg_ok else {}
    if not cfg_ok:
        errors.append(f"Ours config missing: {ours_cfg_path}")
    else:
        frontend = cfg.get("frontend", {})
        shape = cfg.get("shape_bank", {})
        renderer = cfg.get("renderer", {})
        _check(checks, "same_mask_file", frontend.get("mask_source") == "file", frontend.get("mask_source"))
        if frontend.get("mask_source") != "file":
            warnings.append("Final benchmark should use frontend.mask_source=file or fixed cache masks.")
        for key in ("sam_backend", "dino_backend", "depth_backend"):
            if frontend.get(key) == "stub":
                msg = f"{key}=stub; debug only."
                (errors if strict else warnings).append(msg)
        if shape.get("backend") == "toy" or (shape.get("backend") == "auto" and not shape.get("root")):
            msg = "Shape bank may fallback to ToyShapeBank; debug only."
            (errors if strict else warnings).append(msg)
        if renderer.get("backend") == "soft" or renderer.get("fallback_to_soft", True):
            msg = "Renderer may use SoftGaussianRenderer; report clearly or use strict final renderer."
            (errors if strict else warnings).append(msg)
        try:
            rend = detect_renderer_capability(cfg)
            summary["renderer"] = rend
        except Exception as exc:
            warnings.append(f"renderer capability check failed: {exc}")

    cache = profile.get("frontend_cache", {})
    if cache.get("use_cache"):
        cm = cache.get("cache_manifest")
        ok = bool(cm and Path(cm).exists())
        _check(checks, "frontend_cache_manifest", ok, cm or "")
        if not ok:
            errors.append(f"frontend cache manifest missing: {cm}")
        elif cache.get("validate_cache"):
            val = validate_frontend_cache_manifest(cm)
            _check(checks, "frontend_cache_valid", bool(val.get("valid")), f"valid={val.get('valid')}")
            if not val.get("valid"):
                errors.append("frontend cache validation failed")

    catalog_path = profile.get("baselines", {}).get("method_catalog") or profile.get("comparison", {}).get("method_catalog")
    if catalog_path:
        _check(checks, "method_catalog", Path(catalog_path).exists(), catalog_path)
        if Path(catalog_path).exists():
            methods = load_method_catalog(catalog_path)
            external_enabled = [m for m in methods if m.source == "external" and m.enabled]
            if external_enabled:
                warnings.append(f"Enabled external baselines need provided outputs: {[m.name for m in external_enabled]}")
            elif profile.get("readiness", {}).get("warn_if_missing_external_baselines", True):
                warnings.append("External baselines are disabled/missing; final paper comparison may be incomplete.")

    if out_dir is not None:
        for key in ("tables", "report"):
            target = Path(out_dir) / profile.get(key, {}).get("output_dir", key)
            target.mkdir(parents=True, exist_ok=True)
            _check(checks, f"{key}_writable", target.exists(), target)

    ready = len(errors) == 0
    return {"ready": ready, "strict": bool(strict), "errors": errors, "warnings": warnings, "checks": checks, "summary": summary}


def save_final_readiness_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(report, out / "final_readiness.json")
    lines = ["# Final Paper Readiness", "", f"Ready: `{report.get('ready')}`", f"Strict: `{report.get('strict')}`", ""]
    lines += ["## Errors"] + [f"- {e}" for e in report.get("errors", [])] + [""]
    lines += ["## Warnings"] + [f"- {w}" for w in report.get("warnings", [])] + [""]
    lines += ["## Checklist", "| Check | OK | Message |", "|---|---:|---|"]
    for c in report.get("checks", []):
        lines.append(f"| {c.get('name')} | {c.get('ok')} | {c.get('message')} |")
    (out / "final_readiness.md").write_text("\n".join(lines), encoding="utf-8")
