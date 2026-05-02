from __future__ import annotations

from pathlib import Path

from shapesplat.experiments.paper_runner import load_paper_profile
from shapesplat.datasets.benchmark.validator_v2 import validate_benchmark_v2


def check_paper_ready(profile_path: str | Path, out_dir: str | Path) -> dict:
    """检查 paper profile 是否具备基本运行条件。

    这里做的是论文实验前的轻量 readiness check：只验证配置、脚本、
    manifest 和输出目录等协议层条件，不强制要求真实 SAM/DINO/Depth
    或外部 baseline 已经安装。
    """

    path = Path(profile_path)
    checks: list[dict] = []
    warnings: list[str] = []
    errors: list[str] = []
    if not path.exists():
        errors.append(f"profile missing: {path}")
        return {"ready": False, "checks": checks, "warnings": warnings, "errors": errors}

    profile = load_paper_profile(path)
    checks.append({"name": "profile_exists", "ok": True, "path": str(path)})

    for key, value in (profile.get("configs") or {}).items():
        ok = Path(value).exists()
        checks.append({"name": f"config_{key}", "ok": ok, "path": value})
        if not ok:
            errors.append(f"config missing: {value}")

    for key in ["config", "ablations"]:
        if profile.get(key):
            ok = Path(profile[key]).exists()
            checks.append({"name": key, "ok": ok, "path": profile[key]})
            if not ok:
                errors.append(f"{key} missing: {profile[key]}")

    manifest = profile.get("manifest")
    if profile.get("benchmark_manifest"):
        manifest = profile.get("benchmark_manifest")
    if manifest:
        manifest_ok = Path(manifest).exists() or bool(profile.get("create_if_missing"))
        checks.append({"name": "manifest", "ok": manifest_ok, "path": manifest})
        if not manifest_ok:
            errors.append(f"manifest missing: {manifest}")
        elif Path(manifest).exists() and profile.get("benchmark_schema") == "v2":
            # benchmark v2 readiness 会检查协议和可选字段；optional GT 缺失只作为 warning。
            bench = validate_benchmark_v2(manifest, check_optional_gt=True, check_cache=bool(profile.get("use_frontend_cache")))
            checks.append({"name": "benchmark_v2", "ok": bench["valid"], "num_rows": bench["num_rows"], "num_failed": bench["num_failed"]})
            warnings.extend(bench.get("warnings", []))
            for row in bench.get("rows", []):
                warnings.extend([f"{row.get('image_id')}: {w}" for w in row.get("warnings", [])[:3]])
            if not bench["valid"]:
                errors.append(f"benchmark v2 invalid: {manifest}")

    scripts = [
        "scripts/run_comparison.py",
        "scripts/run_ablation.py",
        "scripts/run_stress_benchmark.py",
        "scripts/run_edit_dataset.py",
        "scripts/run_baseline_dataset.py",
        "scripts/generate_report.py",
    ]
    for script in scripts:
        ok = Path(script).exists()
        checks.append({"name": script, "ok": ok})
        if not ok:
            errors.append(f"script missing: {script}")

    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        probe = Path(out_dir) / ".paper_ready_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append({"name": "out_writable", "ok": True, "path": str(out_dir)})
    except OSError as exc:
        checks.append({"name": "out_writable", "ok": False, "path": str(out_dir)})
        errors.append(f"output directory is not writable: {exc}")

    # 这些 warning 是正式实验前的提醒，不阻止 debug/stub pipeline 运行。
    warnings.append("soft renderer is useful for workflow validation but is not final high-quality CUDA 3DGS")
    warnings.append("optional geometry metrics require GT point clouds and are not required for minimal/stub profiles")
    warnings.append("external baseline entries are templates unless user provides real repos/checkpoints")
    return {"ready": len(errors) == 0, "checks": checks, "warnings": warnings, "errors": errors}
