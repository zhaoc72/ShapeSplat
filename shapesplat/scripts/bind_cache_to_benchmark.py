from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.datasets.benchmark.cache_binding import bind_frontend_cache_to_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind frontend cache manifest to benchmark v2 manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()
    out = bind_frontend_cache_to_benchmark(args.manifest, args.cache_manifest, args.out_manifest)
    print(f"manifest with cache: {out}")


if __name__ == "__main__":
    main()

