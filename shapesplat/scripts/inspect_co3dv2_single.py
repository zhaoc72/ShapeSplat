from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.datasets.converters.co3dv2_single import inspect_co3dv2_single


def _markdown(summary: dict) -> str:
    lines = [
        "# CO3Dv2 Single Structure",
        "",
        f"- root: `{summary.get('root')}`",
        f"- exists: {summary.get('exists')}",
        f"- categories: {summary.get('num_categories')}",
        "",
    ]
    for cat in summary.get("categories", []):
        lines.append(f"## {cat['category']}")
        lines.append(f"- sequences: {cat['num_sequences']}")
        lines.append(f"- frame_annotations.jgz: {cat['has_frame_annotations']}")
        lines.append(f"- sequence_annotations.jgz: {cat['has_sequence_annotations']}")
        lines.append(f"- set_lists: {cat['has_set_lists']}")
        lines.append(f"- eval_batches: {cat['has_eval_batches']}")
        for seq in cat.get("sample_sequences", []):
            lines.append(f"- {seq['sequence']}: images={seq['num_images']} masks={seq['num_masks']} depths={seq['num_depths']} pointcloud={seq['has_pointcloud_ply']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local CO3Dv2 single subset structure.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/inspect_co3dv2_single")
    parser.add_argument("--max-categories", type=int, default=5)
    parser.add_argument("--max-sequences", type=int, default=5)
    args = parser.parse_args()
    # 中文注释：转换前结构检查，不依赖 CO3D package / PyTorch3D。
    summary = inspect_co3dv2_single(args.root, args.max_categories, args.max_sequences)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "co3dv2_single_structure.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "co3dv2_single_structure.md").write_text(_markdown(summary), encoding="utf-8")
    print(f"root: {summary['root']}")
    print(f"exists: {summary['exists']} categories: {summary['num_categories']}")
    print(f"report: {(out / 'co3dv2_single_structure.json').resolve()}")


if __name__ == "__main__":
    main()
