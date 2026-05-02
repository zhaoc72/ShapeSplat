from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cache.validate_cache import validate_frontend_cache_manifest
from shapesplat.config import load_config
from shapesplat.experiments.co3dv2_real_frontend import apply_dinov3_cli_overrides, check_checkpoint_path
from shapesplat.frontend.dinov3_dependency_check import check_dinov3_dependencies
from shapesplat.runtime.cli import prepare_runtime_for_run
from shapesplat.cache.frontend_cache import write_frontend_cache_manifest
from shapesplat.cache.frontend_cache_runner import cache_frontend_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache CO3Dv2 real frontend outputs with file masks and DINOv3 descriptors.")
    parser.add_argument("--config", default="configs/co3dv2_real_frontend_debug.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-cache", required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--save-dino-features", action="store_true")
    parser.add_argument("--check-deps-first", action="store_true")
    args = parser.parse_args()

    runtime_overrides = {"device": args.device, "require_cuda_for_experiments": args.device == "cuda"} if args.device else None
    cfg = load_config(args.config, runtime_overrides=runtime_overrides)
    apply_dinov3_cli_overrides(cfg, device=args.device)
    if args.require_cuda:
        cfg.setdefault("runtime", {})["device"] = "cuda"
        cfg["runtime"]["require_cuda_for_experiments"] = True
        cfg["device"] = "cuda"
    prepare_runtime_for_run(cfg, args.out_cache, save_summary=True)
    deps = check_dinov3_dependencies()
    if args.check_deps_first and not deps["ok"]:
        out = Path(args.out_cache)
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "num_success": 0,
            "reason": "missing_dinov3_dependencies",
            "dependencies": deps,
            "message": "No frontend cache entries were created. Check DINOv3 dependencies and checkpoint loading.",
        }
        (out / "co3dv2_real_frontend_cache_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(1)
    check_checkpoint_path(cfg.get("frontend", {}).get("dino_checkpoint"), allow_missing=False)

    rows, records = cache_frontend_outputs(
        args.config,
        args.manifest,
        args.out_cache,
        max_images=args.max_images,
        save_dino_features=args.save_dino_features,
        mask_source="file",
    )
    manifest_path = args.write_manifest or str(Path(args.out_cache) / "cache_manifest.csv")
    write_frontend_cache_manifest(records, manifest_path)
    report = {"rows": rows, "cache_manifest": manifest_path, "num_success": sum(r.get("status") == "success" for r in rows), "dependencies": deps}
    if report["num_success"] == 0:
        report["reason"] = "no_successful_frontend_cache_entries"
        report["message"] = "No frontend cache entries were created. Check DINOv3 dependencies and checkpoint loading."
    if args.validate:
        report["validation"] = validate_frontend_cache_manifest(manifest_path)
        (Path(args.out_cache) / "cache_validation.json").write_text(json.dumps(report["validation"], indent=2, ensure_ascii=False), encoding="utf-8")
    (Path(args.out_cache) / "co3dv2_real_frontend_cache_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
