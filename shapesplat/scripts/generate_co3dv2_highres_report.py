from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.experiments.co3dv2_highres_report import generate_co3dv2_highres_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CO3Dv2 high-resolution diagnostic report.")
    parser.add_argument("--root", default="outputs/ours_co3dv2_vits16_highres")
    parser.add_argument("--out", default="outputs/ours_co3dv2_vits16_highres/report")
    parser.add_argument("--title", default="CO3Dv2 High-Resolution Diagnostics")
    args = parser.parse_args()
    manifest = generate_co3dv2_highres_report(args.root, args.out, title=args.title)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
