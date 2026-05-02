from __future__ import annotations

import importlib
from pathlib import Path

from shapesplat.experiments.orchestrator import load_preset


def check_file_exists(path, required: bool = True) -> dict:
    ok = bool(path) and Path(path).exists()
    status = "ok" if ok else ("error" if required else "warning")
    return {"name": "file_exists", "path": str(path), "status": status, "message": "" if ok else f"missing: {path}"}


def check_config_exists(path) -> dict:
    return {**check_file_exists(path, required=True), "name": "config_exists"}


def check_manifest(path) -> dict:
    return {**check_file_exists(path, required=True), "name": "manifest_exists"}


def check_python_imports() -> dict:
    """检查 minimal pipeline 所需核心 import；真实 SAM/DINO/Depth 不在此强制检查。"""

    required = ["torch", "numpy", "PIL", "yaml", "shapesplat"]
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    return {"name": "python_imports", "status": "ok" if not missing else "error", "missing": missing}


def check_outputs_writable(out_dir) -> dict:
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"name": "outputs_writable", "path": str(out), "status": "ok"}
    except Exception as exc:
        return {"name": "outputs_writable", "path": str(out), "status": "error", "message": str(exc)}


def check_experiment_ready(preset_path: str | Path, out_dir: str | Path, context: dict) -> dict:
    """运行前 sanity check：只检查可选输入和核心依赖，不检查 optional real backends。"""

    checks = []
    errors = []
    warnings = []
    checks.append(check_file_exists(preset_path, required=True))
    try:
        plan = load_preset(preset_path)
        checks.append({"name": "preset_load", "status": "ok", "num_steps": len(plan.steps)})
    except Exception as exc:
        checks.append({"name": "preset_load", "status": "error", "message": str(exc)})
    for cfg in ["configs/minimal.yaml", "configs/comparison_minimal.yaml", "configs/same_mask.yaml"]:
        if Path(cfg).exists():
            checks.append(check_config_exists(cfg))
    if context.get("manifest"):
        checks.append(check_manifest(context["manifest"]))
    if context.get("input"):
        checks.append(check_file_exists(context["input"], required=True))
    if context.get("mask"):
        checks.append(check_file_exists(context["mask"], required=True))
    checks.append(check_outputs_writable(out_dir))
    checks.append(check_python_imports())
    for check in checks:
        if check.get("status") == "error":
            errors.append(check.get("message") or check.get("name"))
        elif check.get("status") == "warning":
            warnings.append(check.get("message") or check.get("name"))
    return {"ready": len(errors) == 0, "checks": checks, "warnings": warnings, "errors": errors}

