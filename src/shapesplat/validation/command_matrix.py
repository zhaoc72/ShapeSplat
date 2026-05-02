from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_command_matrix(path: str | Path) -> list[dict]:
    """读取 artifact 命令矩阵。

    command_matrix 是最终交付前的可执行清单：每条命令记录用途分组、
    是否必跑、命令参数和预期输出。
    """

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    commands = data.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError(f"command matrix must contain a commands list: {path}")
    return commands


def _resolve_command(command: list[str], context: dict) -> list[str]:
    resolved = []
    for i, part in enumerate(command):
        value = str(part).format(**context)
        if i == 0 and value.lower() in {"python", "python.exe"}:
            value = sys.executable
        resolved.append(value)
    return resolved


def run_command_entry(entry: dict, context: dict, dry_run: bool = False) -> dict:
    """执行单条命令并检查 expected_outputs。

    不使用 shell=True，保证 Windows PowerShell 和 CI 中行为一致。
    """

    out_root = Path(context.get("validation_out", "outputs/validation"))
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    name = entry.get("name", "unnamed")
    command = _resolve_command(entry.get("command", []), context)
    start = datetime.now()
    row = {
        "name": name,
        "group": entry.get("group", ""),
        "required": bool(entry.get("required", False)),
        "command": command,
        "start_time": start.isoformat(),
    }
    if dry_run:
        row.update({"status": "dry_run", "return_code": None})
    else:
        env = os.environ.copy()
        src = str((Path.cwd() / "src").resolve())
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, env=env)
        stdout_path = logs / f"{name}_stdout.txt"
        stderr_path = logs / f"{name}_stderr.txt"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        row.update(
            {
                "return_code": result.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "status": "success" if result.returncode == 0 else "failed",
            }
        )

    missing = []
    if not dry_run and row["status"] == "success":
        for expected in entry.get("expected_outputs", []) or []:
            if not Path(str(expected).format(**context)).exists():
                missing.append(expected)
        if missing:
            row["status"] = "failed"
    row["missing_outputs"] = missing
    row["end_time"] = datetime.now().isoformat()
    return row


def run_command_matrix(
    path: str | Path,
    groups: list[str] | None = None,
    dry_run: bool = False,
    stop_on_failure: bool = True,
    context: dict | None = None,
) -> list[dict]:
    """按分组运行命令矩阵，返回每条命令的执行结果。"""

    entries = load_command_matrix(path)
    wanted = set(groups or [])
    context = {"validation_out": "outputs/validation", **(context or {})}
    rows = []
    for entry in entries:
        if wanted and entry.get("group") not in wanted:
            continue
        row = run_command_entry(entry, context, dry_run=dry_run)
        rows.append(row)
        if row["status"] == "failed" and stop_on_failure and entry.get("required", False):
            break
    out = Path(context.get("validation_out", "outputs/validation"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "command_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows
