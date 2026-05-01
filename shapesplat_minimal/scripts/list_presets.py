from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.orchestrator import load_preset


def list_presets(preset_dir: str | Path = ROOT / "configs" / "presets") -> list[dict]:
    """列出可用实验 preset，方便用户查看统一入口支持的任务。"""

    rows = []
    for path in sorted(Path(preset_dir).glob("*.yaml")):
        try:
            plan = load_preset(path)
            rows.append({"preset": path.stem, "description": plan.description, "default_out": plan.default_out, "num_steps": len(plan.steps)})
        except Exception as exc:
            rows.append({"preset": path.stem, "description": f"ERROR: {exc}", "default_out": "", "num_steps": 0})
    return rows


def main() -> None:
    rows = list_presets()
    cols = ["preset", "description", "default_out", "num_steps"]
    widths = {c: len(c) for c in cols}
    for row in rows:
        for col in cols:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for row in rows:
        print(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()

