from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.cache.attach import apply_frontend_cache_config, attach_cache_to_dataset
from shapesplat.config import load_config
from shapesplat.datasets.image_dataset import build_dataset_from_manifest
from shapesplat.experiments.comparison_runner import run_comparison_dataset
from shapesplat.experiments.comparison_summary import save_comparison_summary
from shapesplat.evaluation.report import print_metrics
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.utils.comparison_visualization import make_qualitative_index
from shapesplat.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-mask comparison: Ours + dummy baselines.")
    parser.add_argument("--config", default="configs/comparison_minimal.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/comparison_run")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-ours", action="store_true")
    parser.add_argument("--no-dummy-baselines", action="store_true")
    parser.add_argument("--run-independent-gaussian", action="store_true")
    parser.add_argument("--ours-output-dir", default=None, help="读取已运行的 Ours benchmark 输出，避免在 comparison 中重新训练 Ours")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--use-frontend-cache", action="store_true", help="从 cached frontend outputs 读取 masks/descriptors/depth")
    parser.add_argument("--frontend-cache-root", default=None, help="frontend cache 根目录，按 image_id 查找子目录")
    parser.add_argument("--frontend-cache-manifest", default=None, help="frontend cache manifest CSV，用 image_id 映射 cache_dir")
    parser.add_argument("--save-frontend-cache", action="store_true", help="运行前端后保存 cache，便于后续复用")
    parser.add_argument("--frontend-cache-out", default=None, help="保存 frontend cache 的根目录")
    parser.add_argument("--no-run-metadata", action="store_true", help="不写入 run_info / registry 元数据")
    parser.add_argument("--registry", default="runs/run_registry.jsonl", help="全局 run registry 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # comparison runner 的默认协议是 same-mask；这里显式设置 file，避免 SAM proposal 干扰公平比较。
    cfg.setdefault("frontend", {})
    cfg["frontend"]["mask_source"] = "file"
    apply_frontend_cache_config(
        cfg,
        use_cache=args.use_frontend_cache,
        cache_root=args.frontend_cache_root,
        cache_manifest=args.frontend_cache_manifest,
        save_cache=args.save_frontend_cache,
        cache_out=args.frontend_cache_out,
    )
    seed_everything(int(cfg["seed"]))

    dataset = build_dataset_from_manifest(args.manifest, image_size=int(cfg["image"]["size"]))
    # cached frontend outputs 通过 image_id 绑定到 dataset record，旧 manifest 没有该列时保持原行为。
    cache_cfg = cfg.get("frontend_cache", {})
    attach_cache_to_dataset(
        dataset,
        cache_manifest=args.frontend_cache_manifest or cache_cfg.get("cache_manifest"),
        cache_root=args.frontend_cache_root or args.frontend_cache_out or cache_cfg.get("cache_root"),
    )
    rows = run_comparison_dataset(
        dataset,
        cfg,
        args.out,
        max_images=args.max_images,
        skip_existing=args.skip_existing,
        run_ours=not args.no_ours,
        run_dummy_baselines=not args.no_dummy_baselines,
        run_independent_gaussian=args.run_independent_gaussian,
        save_visuals=not args.no_visuals,
        save_checkpoint=args.save_checkpoint,
        save_frontend_cache=args.save_frontend_cache,
        ours_output_dir=args.ours_output_dir,
    )
    summary = save_comparison_summary(rows, args.out)
    make_qualitative_index(args.out, Path(args.out) / "qualitative_index.md")
    print_metrics(summary)
    print(f"comparison outputs saved to: {Path(args.out).resolve()}")
    if not args.no_run_metadata:
        try:
            # comparison run 记录 same-mask manifest 和 method-wise summary。
            finalize_run_outputs(
                out_dir=args.out,
                config_path=args.config,
                run_type="comparison",
                manifest_path=args.manifest,
                registry_path=args.registry,
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
