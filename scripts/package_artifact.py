from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.validation.artifact import create_artifact_package, make_artifact_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a lightweight ShapeSplat++ artifact zip.")
    parser.add_argument("--out", default="dist/shapesplat_artifact.zip")
    parser.add_argument("--include-examples", action="store_true")
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--include-docs", action="store_true")
    args = parser.parse_args()

    # artifact 包不包含 outputs/runs/checkpoints；真实模型权重需单独配置。
    path = create_artifact_package(
        args.out,
        include_docs=args.include_docs,
        include_tests=args.include_tests,
        include_examples=args.include_examples,
    )
    manifest = make_artifact_manifest(".")
    manifest_path = Path(args.out).parent / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"artifact: {path}")
    print(f"manifest: {manifest_path}")
    print(f"size: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
