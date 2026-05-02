from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from shapesplat.config import load_config
from shapesplat.datasets.benchmark.validator_v2 import save_benchmark_v2_validation, validate_benchmark_v2
from shapesplat.reconstruction.ours_runner import run_ours_benchmark
from shapesplat.reporting.final_report import generate_final_report


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _maybe_apply_debug_iteration_cap(cfg: dict) -> None:
    """在 fallback debug 环境中压低迭代数，避免 CO3Dv2 烟测变成超长训练。"""
    ours = cfg.get("ours", {})
    if not ours.get("auto_debug_iteration_cap", False):
        return
    cache = cfg.get("frontend_cache", {})
    shape = cfg.get("shape_bank", {})
    renderer = cfg.get("renderer", {})
    fallback_debug = (
        not (cache.get("cache_manifest") or cache.get("cache_root"))
        and not shape.get("root")
        and renderer.get("fallback_to_soft", True)
    )
    if not fallback_debug:
        return
    for key, value in ours.get("debug_iteration_cap", {}).items():
        if key in cfg.get("training", {}):
            cfg["training"][key] = int(value)
    print("warning: applying CO3Dv2 diagnostic debug iteration cap because fallback components are active.")


def run_co3dv2_diagnostics(
    manifest: str | Path,
    config: str | Path,
    out: str | Path,
    max_images: int | None = None,
    use_frontend_cache: bool = False,
    frontend_cache_manifest: str | Path | None = None,
    run_editing: bool = False,
    generate_report: bool = False,
) -> dict:
    """运行 CO3Dv2 single real-image diagnostics。

    CO3Dv2 single subset 通常是 object-centric 单 foreground，可用于真实图片诊断
    和 single visible-mask benchmark；这里不把它描述为多物体主 benchmark。
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    limit = 20 if max_images is None else max_images
    val = validate_benchmark_v2(manifest, max_rows=limit, check_optional_gt=False, check_cache=use_frontend_cache)
    save_benchmark_v2_validation(val, out / "validation")

    cfg = load_config(config)
    _maybe_apply_debug_iteration_cap(cfg)
    rows = run_ours_benchmark(
        manifest,
        cfg,
        out / "ours",
        max_images=limit,
        use_frontend_cache=use_frontend_cache,
        frontend_cache_manifest=frontend_cache_manifest,
        save_checkpoint=False,
    )

    outputs: dict[str, str] = {
        "validation": str(out / "validation"),
        "ours": str(out / "ours"),
    }
    editing_status = "skipped"
    if run_editing:
        # 编辑诊断复用已有 CLI，避免 CO3Dv2 接入层重复实现 editing runner。
        edit_cmd = [
            sys.executable,
            "scripts/run_edit_dataset.py",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--out",
            str(out / "editing"),
            "--max-images",
            str(limit),
            "--max-objects",
            "1",
            "--no-run-metadata",
        ]
        completed = subprocess.run(edit_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
        editing_status = "success" if completed.returncode == 0 else "failed"
        (out / "editing_command_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (out / "editing_command_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        outputs["editing"] = str(out / "editing")

    if generate_report:
        # CO3Dv2 single 默认不报告无明确对齐协议的 geometry 指标。
        generate_final_report(out, out / "report", title="ShapeSplat++ CO3Dv2 Diagnostics")
        outputs["report"] = str(out / "report")

    status = "success" if val.get("valid", False) else "validation_failed"
    return {
        "status": status,
        "manifest": str(manifest),
        "out": str(out),
        "num_images": len(rows),
        "num_rows": len(rows),
        "editing_status": editing_status,
        "validation": val,
        "outputs": outputs,
    }
