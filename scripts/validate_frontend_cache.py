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

from shapesplat.cache.frontend_cache import build_cache_manifest_from_root, write_frontend_cache_manifest
from shapesplat.cache.validate_cache import validate_frontend_cache_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frontend cache directories.")
    parser.add_argument("--cache-root", default="outputs/frontend_cache")
    parser.add_argument("--cache-manifest", default=None)
    parser.add_argument("--out", default="outputs/frontend_cache_validation")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.cache_manifest) if args.cache_manifest else out / "cache_manifest_from_root.csv"
    if args.cache_manifest is None:
        records = build_cache_manifest_from_root(args.cache_root)
        write_frontend_cache_manifest(records, manifest)
    report = validate_frontend_cache_manifest(manifest)
    (out / "cache_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "cache_validation.csv", "w", encoding="utf-8", newline="") as f:
        fields = ["image_id", "valid", "num_masks", "descriptor_dim", "height", "width", "errors", "warnings", "cache_dir"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            r = dict(row)
            r["errors"] = "|".join(r.get("errors", []))
            r["warnings"] = "|".join(r.get("warnings", []))
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"num_valid: {report['num_valid']} num_failed: {report['num_failed']}")
    for warning in report.get("warnings", []):
        print(f"warning: {warning}")
    for error in report.get("errors", []):
        print(f"error: {error}")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
