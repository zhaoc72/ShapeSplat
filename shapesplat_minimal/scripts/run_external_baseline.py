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
from shapesplat.baselines.external_runner import run_external_baseline_for_image
from shapesplat.config import load_config
from shapesplat.data.image_io import load_image
from shapesplat.frontend.file_mask_loader import load_mask_file


def _load_external_cfg(path: str | Path, adapter: str) -> tuple[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    runner = data.get("runner", {})
    for item in data.get("external_baselines", []):
        if item.get("name") == adapter or item.get("adapter") == adapter:
            cfg = {**runner, **item}
            return item.get("adapter", adapter), cfg
    raise KeyError(f"Adapter config not found: {adapter}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one external baseline adapter on one same-mask image.")
    parser.add_argument("--config", default="configs/same_mask.yaml")
    parser.add_argument("--external-config", default="configs/external_baselines.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-id", default="image")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image = load_image(args.input, size=int(cfg["image"]["size"]))
    mask_set = load_mask_file(args.mask, image_hw=image.shape[-2:], cfg=cfg)
    input_spec = export_baseline_inputs(image, mask_set.masks, Path(args.out) / "inputs", args.image_id)
    adapter_name, adapter_cfg = _load_external_cfg(args.external_config, args.adapter)
    row = run_external_baseline_for_image(adapter_name, input_spec, args.out, adapter_cfg, dry_run=args.dry_run)
    print(f"status: {row.get('status')}")
    for key in sorted(row):
        if key not in {"status"}:
            print(f"{key}: {row[key]}")


if __name__ == "__main__":
    main()

