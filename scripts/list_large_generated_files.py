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

from shapesplat.cleanup.rules import load_cleanup_rules


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def list_large_files(root: str | Path, min_mb: float, rules: dict) -> list[dict]:
    root_path = Path(root).resolve()
    protected = [(root_path / p).resolve() for p in rules.get("protected_paths", [])]
    rows = []
    threshold = int(float(min_mb) * 1024 * 1024)
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = int(path.stat().st_size)
        except OSError:
            continue
        if size < threshold:
            continue
        rows.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 3),
                "protected": any(path.resolve() == p or _is_under(path, p) for p in protected),
            }
        )
    return sorted(rows, key=lambda r: r["size_bytes"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="List large generated files without deleting anything.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--min-mb", type=float, default=10)
    parser.add_argument("--out", default="outputs/large_files_report")
    parser.add_argument("--config", default="configs/cleanup_generated_artifacts.yaml")
    args = parser.parse_args()
    rules = load_cleanup_rules(args.config)
    rows = list_large_files(args.root, args.min_mb, rules)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "large_files.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "large_files.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["path", "size_bytes", "size_mb", "protected"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"large files: {len(rows)}")
    print(f"report saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
