from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.experiments.co3dv2_real_frontend import (
    apply_dinov3_cli_overrides,
    check_checkpoint_path,
    descriptor_stats,
    make_center_mask,
    save_feature_norm_image,
    save_missing_report,
)
from shapesplat.frontend.dinov3_dependency_check import check_dinov3_dependencies
from shapesplat.frontend.dino_backend import build_dino_backend
from shapesplat.runtime.memory import get_gpu_memory_info


def check_dinov3_weights(config: str, out: str, checkpoint: str | None = None, model_name: str | None = None, input_path: str | None = None, allow_missing: bool = False, device: str | None = None, check_deps_only: bool = False) -> dict:
    """手动检查本地 DINOv3 权重；pytest 不会调用真实加载路径。"""
    runtime_overrides = {"device": device, "require_cuda_for_experiments": device == "cuda"} if device else None
    if check_deps_only and runtime_overrides is None:
        runtime_overrides = {"device": "cpu", "require_cuda_for_experiments": False, "allow_cpu_fallback": True}
    if device == "cpu":
        runtime_overrides["allow_cpu_fallback"] = True
        runtime_overrides["require_cuda_for_experiments"] = False
    cfg = load_config(config, runtime_overrides=runtime_overrides)
    apply_dinov3_cli_overrides(cfg, checkpoint=checkpoint, model_name=model_name, device=device)
    fcfg = cfg["frontend"]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_report = check_checkpoint_path(fcfg.get("dino_checkpoint"), allow_missing=allow_missing or check_deps_only)
    deps = check_dinov3_dependencies()
    report = {
        "dino_repo_path": fcfg.get("dino_repo_path"),
        "model_name": fcfg.get("dino_model_name"),
        "checkpoint": fcfg.get("dino_checkpoint"),
        "checkpoint_exists": ckpt_report["exists"],
        "device": fcfg.get("dino_device", cfg.get("device")),
        "dependencies": deps,
    }
    if check_deps_only:
        report["status"] = "deps_ok" if deps["ok"] else "missing_dependencies"
        save_missing_report(out_dir, "check_dinov3_report", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not deps["ok"]:
            raise SystemExit(1)
        return report
    if not ckpt_report["exists"]:
        report.update({"status": "missing_checkpoint", "checkpoint_report": ckpt_report})
        save_missing_report(out_dir, "check_dinov3_report", report)
        if allow_missing:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return report
        raise FileNotFoundError(ckpt_report["error"])
    if not deps["ok"]:
        report["status"] = "missing_dependencies"
        save_missing_report(out_dir, "check_dinov3_report", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Suggested install command: {deps['install_command']}")
        raise SystemExit(1)

    image = load_image(input_path, size=int(cfg["image"]["size"])) if input_path and Path(input_path).exists() else make_synthetic_image(int(cfg["image"]["size"]))
    dino = build_dino_backend(cfg)
    try:
        features = dino.extract_dense_features(image)
    except ValueError as exc:
        report.update(
            {
                "status": "failed",
                "extraction_mode": getattr(dino, "last_extraction_mode", None),
                "model_input_hw": list(getattr(dino, "last_model_input_hw", [])),
                "error_message": str(exc),
            }
        )
        (out_dir / "check_dinov3_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if "global image embedding" in str(exc):
            print("DINOv3 returned a global embedding, not dense patch features.")
            print("ShapeSplat++ needs get_intermediate_layers or forward_features output.")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise
    masks = make_center_mask(features.shape[-2], features.shape[-1], device=features.device)
    desc = dino.pool_descriptors(features, masks)
    stats = descriptor_stats(desc)
    report.update(
        {
            "status": "success",
            "extraction_mode": getattr(dino, "last_extraction_mode", None),
            "model_input_hw": list(getattr(dino, "last_model_input_hw", [])),
            "feature_shape": list(features.shape),
            "descriptor_shape": list(desc.shape),
            "descriptor_stats": stats,
            "gpu_memory": get_gpu_memory_info(getattr(dino, "device", features.device)),
            "error_message": None,
        }
    )
    save_feature_norm_image(features, out_dir / "feature_norm.png")
    (out_dir / "descriptor_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "check_dinov3_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local official DINOv3 weights.")
    parser.add_argument("--config", default="configs/co3dv2_real_frontend_debug.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--input", default="examples/test_image.png")
    parser.add_argument("--out", default="outputs/check_dinov3_weights")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--check-deps-only", action="store_true")
    args = parser.parse_args()
    check_dinov3_weights(args.config, args.out, args.checkpoint, args.model_name, args.input, args.allow_missing, args.device, args.check_deps_only)


if __name__ == "__main__":
    main()
