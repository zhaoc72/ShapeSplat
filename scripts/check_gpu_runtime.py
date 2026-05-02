from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.runtime.cli import add_runtime_args, apply_runtime_cli_overrides, runtime_overrides_from_args
from shapesplat.runtime.cuda_check import check_cuda_runtime
from shapesplat.runtime.env import collect_runtime_environment
from shapesplat.runtime.memory import get_gpu_memory_info
from shapesplat.runtime.summary import make_runtime_summary, save_runtime_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Windows / CUDA / RTX 5090 runtime compatibility.")
    parser.add_argument("--config", default="configs/local_windows_rtx5090.yaml")
    parser.add_argument("--out", default="outputs/check_gpu_runtime")
    parser.add_argument("--matmul-size", type=int, default=512)
    add_runtime_args(parser)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, runtime_overrides_from_args(args))
        apply_runtime_cli_overrides(cfg, args)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        report = check_cuda_runtime(cfg, matmul_size=args.matmul_size)
        summary = make_runtime_summary(cfg)
        summary["cuda_runtime"] = report
        summary["environment"] = collect_runtime_environment()
        summary["memory"] = get_gpu_memory_info()
        save_runtime_summary(summary, out)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.require_cuda and report.get("status") != "cuda_ok":
            print("CUDA required but smoke test failed. Check PyTorch CUDA build, NVIDIA driver, and conda env.")
            raise SystemExit(1)
    except Exception as exc:
        # 中文注释：给用户清晰错误，不只甩 traceback。
        print(f"GPU runtime check failed: {exc}")
        print("Suggestion: verify conda env, PyTorch CUDA build, NVIDIA driver, and RTX 5090 / sm_120 support.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
