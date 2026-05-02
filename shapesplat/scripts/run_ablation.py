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
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.evaluation.edit_metrics import compute_edit_metrics
from shapesplat.evaluation.metrics import compute_basic_metrics
from shapesplat.evaluation.report import flatten_metrics, merge_metrics, save_metrics_csv, save_metrics_json
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.optimization.trainer import Trainer
from shapesplat.reproducibility.finalize import finalize_run_outputs
from shapesplat.utils.config_override import apply_overrides, load_ablation_file
from shapesplat.utils.logging import save_json
from shapesplat.utils.seed import seed_everything
from shapesplat.utils.visualization import save_input_with_mask_overlay, save_mask_grid, save_render_outputs


SUMMARY_COLUMNS = [
    "name",
    "InstIoU_mean",
    "IsoIoU_mean",
    "AttrAcc",
    "AttrPurity_mean",
    "Leakage",
    "ForegroundAlphaError",
    "CollateralL1",
    "EditLocality",
    "DeletionResidual",
]


def run_one_ablation(base_cfg: dict, experiment: dict, input_path: str | None, out_dir: Path) -> dict:
    """运行一个 ablation experiment，并保存和旧版本一致的输出文件。"""

    name = experiment["name"]
    cfg = apply_overrides(base_cfg, experiment.get("overrides", {}))
    cfg["ablation_name"] = name
    seed_everything(int(cfg["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_input = input_path or cfg["image"].get("input_path")
    if resolved_input:
        image = load_image(resolved_input, size=int(cfg["image"]["size"]))
    else:
        image = make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    front = build_frontend(image, cfg)
    if front.masks.shape[0] == 0:
        raise RuntimeError(f"{name}: front-end produced no masks")
    save_mask_grid(front.masks, out_dir / "masks.png")
    save_input_with_mask_overlay(front.image, front.masks, out_dir / "input_mask_overlay.png")

    trainer = Trainer(front, cfg)
    loss_log = trainer.train()
    render = trainer.render()
    save_render_outputs(render, out_dir)
    save_json(loss_log, out_dir / "loss_log.json")
    trainer.save_checkpoint(out_dir / "checkpoint_minimal.pt")

    metrics = merge_metrics(
        {"name": name},
        compute_basic_metrics(render, front.masks),
        compute_edit_metrics(trainer.scene, trainer.renderer, front, render, cfg, object_id=0),
    )
    save_metrics_json(metrics, out_dir / "metrics.json")
    return metrics


def run_ablation_suite(
    config_path: str | Path,
    ablations_path: str | Path,
    input_path: str | None,
    out: str | Path,
    skip_existing: bool = False,
    max_experiments: int | None = None,
) -> list[dict]:
    """运行 ablations.yaml 中的实验，并生成 summary json/csv。"""

    base_cfg = load_config(config_path)
    experiments = load_ablation_file(ablations_path)
    if max_experiments is not None:
        experiments = experiments[: int(max_experiments)]
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, exp in enumerate(experiments, start=1):
        name = exp["name"]
        exp_dir = out_root / name
        metrics_path = exp_dir / "metrics.json"
        print(f"[{i}/{len(experiments)}] Running ablation: {name}")
        if skip_existing and metrics_path.exists():
            from shapesplat.evaluation.report import load_metrics_json

            metrics = load_metrics_json(metrics_path)
        else:
            metrics = run_one_ablation(base_cfg, exp, input_path, exp_dir)
        flat = flatten_metrics(metrics)
        rows.append({key: flat.get(key, "") for key in SUMMARY_COLUMNS})

    save_metrics_json(rows, out_root / "ablation_summary.json")
    save_metrics_csv(rows, out_root / "ablation_summary.csv")
    print(f"Ablation summary saved to: {out_root.resolve()}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShapeSplat++ minimal ablation suite.")
    parser.add_argument("--config", default="configs/minimal.yaml", help="base config path")
    parser.add_argument("--ablations", default="configs/ablations.yaml", help="ablations.yaml path")
    parser.add_argument("--input", default=None, help="optional RGB input image")
    parser.add_argument("--out", default="outputs/ablations", help="output directory")
    parser.add_argument("--skip-existing", action="store_true", help="reuse existing metrics.json when available")
    parser.add_argument("--max-experiments", type=int, default=None, help="debug: run only first N experiments")
    parser.add_argument("--no-run-metadata", action="store_true", help="不写入 run_info / registry 元数据")
    parser.add_argument("--registry", default="runs/run_registry.jsonl", help="全局 run registry 路径")
    args = parser.parse_args()
    run_ablation_suite(args.config, args.ablations, args.input, args.out, args.skip_existing, args.max_experiments)
    if not args.no_run_metadata:
        try:
            # ablation suite 的元数据记录 base config、输入图像和 ablations 文件。
            finalize_run_outputs(
                out_dir=args.out,
                config_path=args.config,
                run_type="ablation",
                input_path=args.input,
                registry_path=args.registry,
                notes={"ablations": str(args.ablations)},
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
