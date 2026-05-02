from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.integration.smoke import run_real_integration_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local real backend integration smoke test.")
    parser.add_argument("--config", default="configs/local_backend_template.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default="outputs/real_integration_smoke")
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--force-stub-ok", action="store_true", default=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    # smoke test 会允许 auto fallback；真实 backend 缺失不会影响默认项目使用。
    report = run_real_integration_smoke(
        cfg,
        args.input,
        args.out,
        save_cache=args.save_cache,
        run_reconstruction=not args.no_reconstruction,
        force_stub_ok=args.force_stub_ok,
    )
    print(f"status: {report.get('status')}")
    print(f"report: {Path(args.out) / 'integration_report.md'}")


if __name__ == "__main__":
    main()
