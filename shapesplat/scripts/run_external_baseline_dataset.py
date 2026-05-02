from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.baselines.export_inputs import export_baseline_inputs
from shapesplat.baselines.external_runner import run_external_baseline_dataset
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.datasets.manifest import load_manifest
from shapesplat.frontend.file_mask_loader import load_mask_file
from shapesplat.reproducibility.finalize import finalize_run_outputs


def _load_external_cfg(path: str | Path, adapter: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    runner = data.get("runner", {})
    for item in data.get("external_baselines", []):
        if item.get("name") == adapter or item.get("adapter") == adapter:
            return item.get("adapter", adapter), {**runner, **item}
    raise KeyError(f"Adapter config not found: {adapter}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an external baseline adapter on a manifest dataset.")
    parser.add_argument("--config", default="configs/same_mask.yaml")
    parser.add_argument("--external-config", default="configs/external_baselines.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-run-metadata", action="store_true", help="不写入 run_info / registry 元数据")
    parser.add_argument("--registry", default="runs/run_registry.jsonl", help="全局 run registry 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter_name, adapter_cfg = _load_external_cfg(args.external_config, args.adapter)
    records = load_manifest(args.manifest)
    if args.max_images is not None:
        records = records[: max(0, int(args.max_images))]
    input_specs = []
    for record in records:
        image = load_image(record.image_path, size=int(cfg["image"]["size"]))
        mask_path = record.metadata.get("mask_path")
        if not mask_path:
            raise FileNotFoundError(f"record has no mask_path: {record.image_id}")
        masks = load_mask_file(mask_path, image_hw=image.shape[-2:], cfg=cfg).masks
        spec = export_baseline_inputs(image, masks, Path(args.out) / record.image_id / "inputs", record.image_id)
        input_specs.append(spec)
    rows = run_external_baseline_dataset(
        adapter_name,
        input_specs,
        args.out,
        adapter_cfg,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )
    print(f"external baseline rows: {len(rows)}")
    print(f"outputs saved to: {Path(args.out).resolve()}")
    if not args.no_run_metadata:
        try:
            # external baseline dataset runner 记录 manifest 和 adapter 名称。
            finalize_run_outputs(
                out_dir=args.out,
                config_path=args.config,
                run_type="external_baseline_dataset",
                manifest_path=args.manifest,
                registry_path=args.registry,
                notes={"external_config": str(args.external_config), "adapter": str(args.adapter)},
            )
        except Exception as exc:
            print(f"warning: failed to write run metadata: {exc}")


if __name__ == "__main__":
    main()
