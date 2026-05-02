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

from shapesplat.cache.attach import apply_frontend_cache_config
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.editing.suite import run_edit_suite_for_scene
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.utils.seed import seed_everything


def run_edit_demo(
    config_path,
    input_path,
    mask_path,
    out_dir,
    object_id=None,
    ops=None,
    save_visuals=True,
    save_checkpoint=False,
    use_frontend_cache=False,
    frontend_cache_dir=None,
    save_frontend_cache=False,
):
    """单图训练后运行 object-level editing suite。"""

    cfg = load_config(config_path)
    cfg.setdefault("frontend", {})["mask_source"] = "file" if mask_path else cfg.get("frontend", {}).get("mask_source", "sam")
    if mask_path:
        cfg["frontend"]["mask_path"] = str(mask_path)
    apply_frontend_cache_config(cfg, use_cache=use_frontend_cache, cache_root=frontend_cache_dir, save_cache=save_frontend_cache, cache_out=frontend_cache_dir)
    seed_everything(int(cfg.get("seed", 0)))
    image = load_image(input_path, size=int(cfg["image"]["size"]))
    # 单图 demo 中 cache_dir 由 CLI 直接给出；未给出时保持旧的现场前端计算流程。
    front = build_frontend(image, cfg, cache_dir=frontend_cache_dir, use_cache=use_frontend_cache, save_cache=save_frontend_cache)
    trainer = Trainer(front, cfg)
    trainer.train()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        trainer.save_checkpoint(out / "checkpoint_minimal.pt")
    object_ids = [int(object_id)] if object_id is not None else None
    return run_edit_suite_for_scene(trainer.scene, trainer.renderer, front, out, object_ids=object_ids, edit_ops=ops, save_visuals=save_visuals, cfg=cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-image ShapeSplat++ object editing demo.")
    parser.add_argument("--config", default="configs/editing.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--out", default="outputs/edit_demo")
    parser.add_argument("--object-id", type=int, default=None)
    parser.add_argument("--ops", default=None, help="comma-separated ops")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--use-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-root", default=None)
    parser.add_argument("--frontend-cache-manifest", default=None)
    parser.add_argument("--save-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-out", default=None)
    parser.add_argument("--no-run-metadata", action="store_true")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    args = parser.parse_args()
    ops = [x.strip() for x in args.ops.split(",") if x.strip()] if args.ops else None
    cache_dir = args.frontend_cache_out or args.frontend_cache_root
    rows = run_edit_demo(
        args.config,
        args.input,
        args.mask,
        args.out,
        object_id=args.object_id,
        ops=ops,
        save_visuals=not args.no_visuals,
        save_checkpoint=args.save_checkpoint,
        use_frontend_cache=args.use_frontend_cache,
        frontend_cache_dir=cache_dir,
        save_frontend_cache=args.save_frontend_cache,
    )
    print(f"edit rows: {len(rows)}")
    print(f"outputs saved to: {Path(args.out).resolve()}")
    if not args.no_run_metadata:
        try:
            finalize_run_outputs(args.out, args.config, "edit_demo", input_path=args.input, registry_path=args.registry)
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
