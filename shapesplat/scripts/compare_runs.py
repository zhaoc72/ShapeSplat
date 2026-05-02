from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _line_diff(a: str, b: str) -> str:
    import difflib

    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile="run_a", tofile="run_b"))


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two finalized run directories.")
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    a, b, out = Path(args.run_a), Path(args.run_b), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_diff = _line_diff(_read(a / "config_resolved.yaml"), _read(b / "config_resolved.yaml"))
    (out / "config_diff.txt").write_text(config_diff, encoding="utf-8")
    ma, mb = _load_json(a / "metrics_summary.json"), _load_json(b / "metrics_summary.json")
    metrics_diff = {}
    for key in sorted(set(ma) | set(mb)):
        va, vb = ma.get(key), mb.get(key)
        try:
            metrics_diff[key] = {"a": va, "b": vb, "b_minus_a": float(vb) - float(va)}
        except Exception:
            metrics_diff[key] = {"a": va, "b": vb, "changed": va != vb}
    (out / "metrics_diff.json").write_text(json.dumps(metrics_diff, indent=2, ensure_ascii=False), encoding="utf-8")
    ha, hb = _load_json(a / "file_hashes.json"), _load_json(b / "file_hashes.json")
    added = sorted(set(hb) - set(ha))
    removed = sorted(set(ha) - set(hb))
    changed = sorted(k for k in set(ha) & set(hb) if ha[k].get("sha256") != hb[k].get("sha256"))
    hash_diff = {"added": added, "removed": removed, "changed": changed}
    (out / "hash_diff.json").write_text(json.dumps(hash_diff, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# Run Compare Report", "", f"- run_a: `{a}`", f"- run_b: `{b}`", f"- config diff lines: {len(config_diff.splitlines())}", f"- hash added: {len(added)}", f"- hash removed: {len(removed)}", f"- hash changed: {len(changed)}", ""]
    (out / "compare_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"compare report saved to: {out.resolve()}")


if __name__ == "__main__":
    main()

