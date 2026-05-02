from __future__ import annotations

import sys
from pathlib import Path

from shapesplat.config import load_config
from shapesplat.reproducibility.environment import collect_environment_info, try_collect_git_info
from shapesplat.reproducibility.hashing import write_file_hashes
from shapesplat.reproducibility.registry import append_run_registry
from shapesplat.reproducibility.run_info import create_run_info, save_run_info
from shapesplat.reproducibility.snapshot import save_command, save_output_index, save_resolved_config
from shapesplat.utils.metrics_summary import extract_metrics_summary


def current_command() -> str:
    """记录当前 Python 命令，便于复现实验。"""

    return " ".join([sys.executable, *sys.argv])


def finalize_run_outputs(
    out_dir: str | Path,
    config_path: str | None,
    run_type: str,
    input_path: str | None = None,
    manifest_path: str | None = None,
    status: str = "success",
    registry_path: str | Path = "runs/run_registry.jsonl",
    command: str | None = None,
    seed: int | None = None,
    notes: dict | None = None,
) -> dict:
    """为已有输出目录补充可复现 metadata。

    该函数只做后处理：保存 run_info、resolved config、环境、输出索引、hash 和
    registry 摘要。调用方应捕获异常，避免 metadata 失败影响主实验。
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path) if config_path else {}
    if seed is None:
        seed = cfg.get("seed") if isinstance(cfg, dict) else None
    info = create_run_info(
        run_type=run_type,
        output_dir=out,
        command=command or current_command(),
        config_path=config_path,
        input_path=input_path,
        manifest_path=manifest_path,
        seed=seed,
        notes=notes or {},
        status=status,
    )
    save_run_info(info, out / "run_info.json")
    if cfg:
        save_resolved_config(cfg, out / "config_resolved.yaml")
    save_command(info.command, out / "command.txt")
    env = collect_environment_info()
    env["git"] = try_collect_git_info(Path(__file__).resolve().parents[3])
    import json

    with open(out / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    save_output_index(out, out / "output_index.json")
    metrics = extract_metrics_summary(out)
    with open(out / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_file_hashes(out, out / "file_hashes.json")
    append_run_registry(info, registry_path=registry_path, metrics_summary=metrics)
    return {"run_id": info.run_id, "metrics_summary": metrics, "registry_path": str(registry_path)}

