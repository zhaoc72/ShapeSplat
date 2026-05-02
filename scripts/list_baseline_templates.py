from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.baselines.templates.object_centric_templates import OBJECT_CENTRIC_METHODS, make_object_centric_command_template
from shapesplat.baselines.templates.scene_level_templates import SCENE_LEVEL_METHODS, make_scene_level_command_template


def main() -> None:
    rows = []
    for m in OBJECT_CENTRIC_METHODS:
        rows.append((m, "object_centric", make_object_centric_command_template(m)))
    for m in SCENE_LEVEL_METHODS:
        rows.append((m, "scene_level", make_scene_level_command_template(m)))
    print("method | family | command_template")
    print("-------+--------+-----------------")
    for method, family, cmd in rows:
        print(f"{method} | {family} | {cmd}")


if __name__ == "__main__":
    main()
