from __future__ import annotations

import argparse
import subprocess
import sys


QUICK_TESTS = [
    "tests/test_smoke.py",
    "tests/test_real_input.py",
    "tests/test_evaluation.py",
    "tests/test_renderer_backend.py",
    "tests/test_file_masks.py",
    "tests/test_orchestrator.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick or full ShapeSplat++ tests.")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pattern", default=None)
    args = parser.parse_args()

    targets = ["tests"] if args.full else QUICK_TESTS
    if args.pattern:
        targets = [args.pattern]
    # 使用当前 Python 解释器调用 pytest，确保 Windows/conda 环境一致。
    cmd = [sys.executable, "-m", "pytest", *targets, "-v"]
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
