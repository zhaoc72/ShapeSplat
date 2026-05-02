from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentStep:
    name: str
    enabled: bool
    command: list[str]
    cwd: Optional[str] = None
    env: dict = field(default_factory=dict)
    allow_failure: bool = False


@dataclass
class ExperimentPlan:
    name: str
    description: str
    default_out: str
    steps: list[ExperimentStep]
    metadata: dict = field(default_factory=dict)


def load_preset(path: str | Path) -> ExperimentPlan:
    """读取 preset YAML；preset 只描述如何调用已有脚本，不改变算法逻辑。"""

    preset_path = Path(path)
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset not found: {preset_path}")
    with open(preset_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    steps_raw = data.get("steps") or []
    if not steps_raw:
        raise ValueError(f"Preset has no steps: {preset_path}")
    steps = [
        ExperimentStep(
            name=str(item["name"]),
            enabled=bool(item.get("enabled", True)),
            command=[str(x) for x in item.get("command", [])],
            cwd=item.get("cwd"),
            env=item.get("env") or {},
            allow_failure=bool(item.get("allow_failure", False)),
        )
        for item in steps_raw
    ]
    return ExperimentPlan(
        name=str(data.get("name") or preset_path.stem),
        description=str(data.get("description") or ""),
        default_out=str(data.get("default_out") or f"outputs/experiments/{preset_path.stem}"),
        steps=steps,
        metadata=data.get("metadata") or {},
    )


def resolve_placeholders(command: list[str], context: dict) -> list[str]:
    """替换命令中的 {out}/{preset}/{timestamp} 等占位符。"""

    resolved = []
    for part in command:
        value = str(part)
        for key, replacement in context.items():
            value = value.replace("{" + str(key) + "}", "" if replacement is None else str(replacement))
        resolved.append(value)
    return resolved


def _json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _normalize_python_command(command: list[str]) -> list[str]:
    # preset 中写 python 更易读；实际执行时使用当前解释器，避免 Windows PATH/conda 差异。
    if command and command[0].lower() in {"python", "python.exe"}:
        return [sys.executable, *command[1:]]
    return command


def run_experiment_plan(
    plan: ExperimentPlan,
    out_dir: str | Path,
    context: dict,
    dry_run: bool = False,
    stop_on_failure: bool = True,
) -> dict:
    """串行执行实验 plan。

    orchestrator 只是命令编排和日志记录，不直接实现 reconstruction / editing / baseline 算法。
    """

    out = Path(out_dir)
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    _json_dump(asdict(plan), out / "experiment_plan.json")

    step_rows: list[dict] = []
    overall_status = "success"
    for step in plan.steps:
        if not step.enabled:
            step_rows.append({"name": step.name, "status": "skipped", "command": step.command})
            continue
        command = _normalize_python_command(resolve_placeholders(step.command, context))
        stdout_path = logs / f"{step.name}_stdout.txt"
        stderr_path = logs / f"{step.name}_stderr.txt"
        start = time.time()
        row = {
            "name": step.name,
            "command": command,
            "start_time": datetime.now().isoformat(timespec="seconds"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "allow_failure": step.allow_failure,
        }
        print(f"[orchestrator] {step.name}: {' '.join(command)}")
        if dry_run:
            row.update({"status": "dry_run", "return_code": None})
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
        else:
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in step.env.items()})
            proc = subprocess.run(
                command,
                cwd=step.cwd or context.get("project_root"),
                env=env,
                capture_output=True,
                text=True,
                shell=False,
            )
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
            status = "success" if proc.returncode == 0 else "failed"
            row.update({"status": status, "return_code": proc.returncode})
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr)
            if proc.returncode != 0:
                overall_status = "failed"
                if not step.allow_failure and stop_on_failure:
                    row["end_time"] = datetime.now().isoformat(timespec="seconds")
                    row["duration_sec"] = time.time() - start
                    step_rows.append(row)
                    break
        row["end_time"] = datetime.now().isoformat(timespec="seconds")
        row["duration_sec"] = time.time() - start
        step_rows.append(row)

    summary = {
        "name": plan.name,
        "description": plan.description,
        "out_dir": str(out),
        "dry_run": bool(dry_run),
        "status": overall_status if not dry_run else "dry_run",
        "num_steps": len(step_rows),
        "num_success": sum(1 for r in step_rows if r.get("status") == "success"),
        "num_failed": sum(1 for r in step_rows if r.get("status") == "failed"),
        "num_skipped": sum(1 for r in step_rows if r.get("status") == "skipped"),
        "steps": step_rows,
    }
    _json_dump(step_rows, out / "command_log.json")
    _json_dump(summary, out / "run_summary.json")
    return summary
