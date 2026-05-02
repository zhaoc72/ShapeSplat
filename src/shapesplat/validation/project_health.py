from __future__ import annotations

from pathlib import Path

import yaml


def check_required_files() -> dict:
    """检查 artifact 交付所需的核心文件是否存在。"""

    required = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "configs/minimal.yaml",
        "scripts/run_minimal.py",
        "scripts/run_experiment.py",
        "src/shapesplat/__init__.py",
        "tests",
    ]
    checks = [{"path": p, "ok": Path(p).exists()} for p in required]
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def check_configs_loadable() -> dict:
    """遍历 configs 下所有 YAML，确保配置文件语法可读。"""

    checks = []
    for path in sorted(Path("configs").rglob("*.yaml")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            checks.append({"path": str(path), "ok": True})
        except Exception as exc:
            checks.append({"path": str(path), "ok": False, "error": str(exc)})
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def check_scripts_exist() -> dict:
    """检查 scripts 入口是否存在；不执行耗时实验。"""

    scripts = sorted(Path("scripts").glob("*.py"))
    required = ["run_minimal.py", "run_experiment.py", "run_paper_experiments.py", "validate_artifact.py"]
    missing = [name for name in required if not (Path("scripts") / name).exists()]
    return {"ok": not missing, "num_scripts": len(scripts), "missing": missing}


def check_project_health() -> dict:
    """汇总项目健康检查。

    这是轻量工程验收，不替代 pytest；它用于快速发现缺文件、配置损坏
    或脚本入口缺失。
    """

    checks = [
        {"name": "required_files", **check_required_files()},
        {"name": "configs_loadable", **check_configs_loadable()},
        {"name": "scripts_exist", **check_scripts_exist()},
    ]
    errors = []
    for check in checks:
        if not check.get("ok", False):
            errors.append(check["name"])
    warnings = []
    if not Path(".github/workflows/ci.yml").exists():
        warnings.append("CI workflow missing")
    return {"healthy": len(errors) == 0, "checks": checks, "warnings": warnings, "errors": errors}

