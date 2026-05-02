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

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.evaluation.report import print_metrics
from shapesplat.experiments.single_image import run_single_image_experiment
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.runtime.cli import add_runtime_args, apply_runtime_cli_overrides, prepare_runtime_for_run, runtime_overrides_from_args
from shapesplat.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--mask", default=None, help="Optional file mask for same-mask mode.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-run-metadata", action="store_true")
    parser.add_argument("--registry", default="runs/run_registry.jsonl")
    parser.add_argument("--use-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-root", default=None)
    parser.add_argument("--frontend-cache-manifest", default=None)
    parser.add_argument("--save-frontend-cache", action="store_true")
    parser.add_argument("--frontend-cache-out", default=None)
    add_runtime_args(parser)
    return parser.parse_args()


def run_pipeline(
    config_path: str | Path,
    out: str | Path,
    input_path: str | Path | None = None,
    do_eval: bool = False,
    mask_path: str | Path | None = None,
    frontend_cache_dir: str | Path | None = None,
    use_frontend_cache: bool = False,
    save_frontend_cache: bool = False,
    runtime_args=None,
) -> Path:
    """运行单图 minimal pipeline；cache 参数默认关闭，保持旧命令行为。"""

    cfg = load_config(config_path, runtime_overrides_from_args(runtime_args) if runtime_args is not None else None)
    if runtime_args is not None:
        # 中文注释：应用 GPU runtime CLI 覆盖，显式 cuda 不可用时会清晰报错。
        apply_runtime_cli_overrides(cfg, runtime_args)
    if mask_path is not None:
        cfg["frontend"]["mask_source"] = "file"
        cfg["frontend"]["mask_path"] = str(mask_path)
    seed_everything(int(cfg["seed"]))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_runtime_for_run(cfg, out_dir, save_summary=bool(getattr(runtime_args, "runtime_summary", False)))

    resolved_input = input_path or cfg["image"].get("input_path")
    if resolved_input:
        print(f"Using input image: {resolved_input}")
        image = load_image(resolved_input, size=int(cfg["image"]["size"]))
    else:
        print("No input image provided. Using synthetic image.")
        image = make_synthetic_image(int(cfg["image"]["size"]))
    print(f"Mask source: {cfg['frontend'].get('mask_source', 'sam')}")

    metrics = run_single_image_experiment(
        image=image,
        cfg=cfg,
        out_dir=out_dir,
        image_id="minimal",
        save_visuals=True,
        save_checkpoint=True,
        eval_metrics=do_eval,
        frontend_cache_dir=frontend_cache_dir,
        use_frontend_cache=use_frontend_cache,
        save_frontend_cache=save_frontend_cache,
    )
    if do_eval:
        print_metrics(metrics)
    print(f"ShapeSplat++ minimal outputs saved to: {out_dir.resolve()}")
    return out_dir


def main() -> None:
    args = parse_args()
    out_dir = run_pipeline(
        args.config,
        args.out,
        args.input,
        do_eval=args.eval,
        mask_path=args.mask,
        frontend_cache_dir=args.frontend_cache_out or args.frontend_cache_root,
        use_frontend_cache=args.use_frontend_cache,
        save_frontend_cache=args.save_frontend_cache,
        runtime_args=args,
    )
    if not args.no_run_metadata:
        try:
            finalize_run_outputs(
                out_dir=out_dir,
                config_path=args.config,
                run_type="minimal",
                input_path=args.input,
                registry_path=args.registry,
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
