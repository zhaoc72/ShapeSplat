from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def inspect_output_resolution(out: str | Path, max_items: int = 5) -> dict:
    """扫描 Ours per-image 输出，汇总 high-res diagnostics 分辨率字段。"""
    root = Path(out)
    per_image = root / "per_image"
    rows = []
    for diag_path in sorted(per_image.glob("*/diagnostics.json"))[: int(max_items)]:
        data = json.loads(diag_path.read_text(encoding="utf-8"))
        row = {
            "image_id": diag_path.parent.name,
            "original_image_shape": data.get("original_image_shape"),
            "working_image_shape": data.get("working_image_shape"),
            "working_mask_shape": data.get("working_mask_shape"),
            "renderer_image_shape": data.get("renderer_image_shape"),
            "dino_input_size": data.get("dino_input_size"),
            "debug_iteration_cap_applied": data.get("debug_iteration_cap_applied"),
        }
        rows.append(row)
    summary = {"out": str(root), "num_rows": len(rows), "rows": rows}
    root.mkdir(parents=True, exist_ok=True)
    (root / "output_resolution_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(root / "output_resolution_summary.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["image_id", "original_image_shape", "working_image_shape", "working_mask_shape", "renderer_image_shape", "dino_input_size", "debug_iteration_cap_applied"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect per-image output resolution diagnostics.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-items", type=int, default=5)
    args = parser.parse_args()
    report = inspect_output_resolution(args.out, args.max_items)
    for row in report["rows"]:
        print(row)
    print(f"resolution summary saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
