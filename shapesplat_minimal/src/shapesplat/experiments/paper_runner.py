from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from shapesplat.reporting.io import load_json, save_json
from shapesplat.reporting.paper_tables import export_paper_table
from shapesplat.reporting.tables import flatten_summary


def load_paper_profile(path: str | Path) -> dict:
    """读取 paper profile；profile 只描述实验编排，不承载新算法。"""
    with open(path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}
    if "profile" not in profile:
        raise ValueError(f"paper profile missing 'profile': {path}")
    return profile


def _cmd(*parts) -> list[str]:
    return [str(p) for p in parts if p is not None and str(p) != ""]


def _frontend_cache_args(profile: dict) -> list[str]:
    """把 paper profile 中的 frontend_cache 配置翻译成下游 CLI 参数。
    这里只是编排参数，不改变任何算法；默认未开启时返回空列表。
    """

    cache = profile.get("frontend_cache") or {}
    if profile.get("use_frontend_cache"):
        cache = {**cache, "use_cache": True, "cache_manifest": profile.get("frontend_cache_manifest") or cache.get("cache_manifest")}
    if not cache.get("use_cache"):
        return []
    args = ["--use-frontend-cache"]
    if cache.get("cache_manifest"):
        args += ["--frontend-cache-manifest", str(cache["cache_manifest"])]
    if cache.get("cache_root"):
        args += ["--frontend-cache-root", str(cache["cache_root"])]
    return args


def _profile_commands(profile: dict, out_dir: Path) -> list[dict]:
    """把 paper profile 展开为已有脚本命令；这是 orchestration，不改变算法。"""
    p = profile.get("profile")
    max_images = profile.get("max_images")
    cache_args = _frontend_cache_args(profile)
    commands: list[dict] = []
    if profile.get("create_example_dataset"):
        commands.append({"name": "create_example_dataset", "command": _cmd(sys.executable, "scripts/create_example_dataset.py", "--out", "examples/example_dataset", "--num-images", "4", "--size", "128")})
    if profile.get("create_stress_dataset"):
        commands.append({"name": "create_stress_dataset", "command": _cmd(sys.executable, "scripts/create_stress_dataset.py", "--out", "examples/stress_dataset", "--num-per-subset", "2", "--size", "128")})

    def add_main(local_profile: dict, subdir: str = "main_comparison"):
        manifest = local_profile.get("benchmark_manifest") or local_profile.get("manifest", "examples/example_dataset/manifest.csv")
        cmd = _cmd(sys.executable, "scripts/run_comparison.py", "--config", local_profile.get("config", "configs/benchmark_baseline.yaml"), "--manifest", manifest, "--out", out_dir / subdir, "--no-run-metadata")
        if local_profile.get("max_images") is not None:
            cmd += ["--max-images", str(local_profile["max_images"])]
        if local_profile.get("run_independent_gaussian", True):
            cmd.append("--run-independent-gaussian")
        if not local_profile.get("run_dummy_baselines", True):
            cmd.append("--no-dummy-baselines")
        cmd += _frontend_cache_args(local_profile) or cache_args
        commands.append({"name": subdir, "command": cmd})

    def add_ours(local_profile: dict, subdir: str = "final_ours_debug"):
        # Ours core runner 直接面向 benchmark v2 manifest，仍然通过 CLI 编排，避免在 paper layer 重写算法流程。
        manifest = local_profile.get("benchmark_manifest") or local_profile.get("manifest", "data/example_benchmark_v2/manifest.csv")
        cmd = _cmd(sys.executable, "scripts/run_ours_benchmark.py", "--config", local_profile.get("config", "configs/final_ours.yaml"), "--manifest", manifest, "--out", out_dir / subdir, "--no-run-metadata")
        if local_profile.get("max_images") is not None:
            cmd += ["--max-images", str(local_profile["max_images"])]
        if local_profile.get("frontend_cache_manifest"):
            cmd += ["--frontend-cache-manifest", str(local_profile["frontend_cache_manifest"])]
        if local_profile.get("use_frontend_cache"):
            cmd.append("--use-frontend-cache")
        commands.append({"name": subdir, "command": cmd})

    if p == "main_comparison":
        add_main(profile, profile.get("out_name", "main_comparison"))
    elif profile.get("run_ours_benchmark"):
        add_ours(profile, profile.get("out_name", "final_ours_debug"))
    elif p == "final_main":
        # final_main 不重新训练方法，只统一评估已有 method outputs 并导出最终表格。
        subdir = profile.get("out_name", "final_main")
        cmd = _cmd(
            sys.executable,
            "scripts/run_final_comparison.py",
            "--manifest",
            profile.get("benchmark_manifest", profile.get("manifest", "data/example_benchmark_v2/manifest.csv")),
            "--methods",
            profile.get("method_catalog", "configs/method_catalog.yaml"),
            "--outputs-config",
            profile.get("method_outputs", "configs/final_method_outputs.yaml"),
            "--config",
            profile.get("config", "configs/final_ours.yaml"),
            "--out",
            out_dir / subdir,
        )
        if profile.get("max_images") is not None:
            cmd += ["--max-images", str(profile["max_images"])]
        commands.append({"name": "final_comparison", "command": cmd})
        commands.append(
            {
                "name": "export_final_tables",
                "command": _cmd(
                    sys.executable,
                    "scripts/export_final_tables.py",
                    "--summary",
                    out_dir / subdir / "final_method_summary.json",
                    "--out",
                    out_dir / subdir / "tables",
                ),
            }
        )
    elif p == "ablation":
        cmd = _cmd(sys.executable, "scripts/run_ablation.py", "--config", profile.get("config", "configs/minimal.yaml"), "--ablations", profile.get("ablations", "configs/ablations.yaml"), "--input", profile.get("input"), "--out", out_dir / profile.get("out_name", "ablation"), "--no-run-metadata")
        if profile.get("max_experiments") is not None:
            cmd += ["--max-experiments", str(profile["max_experiments"])]
        commands.append({"name": "ablation", "command": cmd})
    elif p == "stress":
        if profile.get("create_if_missing"):
            commands.append({"name": "create_stress_dataset", "command": _cmd(sys.executable, "scripts/create_stress_dataset.py", "--out", "examples/stress_dataset", "--num-per-subset", "2", "--size", "128")})
        manifest = profile.get("benchmark_manifest") or profile.get("manifest", "examples/stress_dataset/manifest.csv")
        cmd = _cmd(sys.executable, "scripts/run_stress_benchmark.py", "--config", profile.get("config", "configs/stress_benchmark.yaml"), "--manifest", manifest, "--out", out_dir / profile.get("out_name", "stress"), "--no-run-metadata")
        if profile.get("max_images") is not None:
            cmd += ["--max-images", str(profile["max_images"])]
        cmd += cache_args
        commands.append({"name": "stress", "command": cmd})
    elif p == "editing":
        manifest = profile.get("benchmark_manifest") or profile.get("manifest", "examples/example_dataset/manifest.csv")
        cmd = _cmd(sys.executable, "scripts/run_edit_dataset.py", "--config", profile.get("config", "configs/editing.yaml"), "--manifest", manifest, "--out", out_dir / profile.get("out_name", "editing"), "--max-objects", profile.get("max_objects", 2), "--no-run-metadata")
        if profile.get("max_images") is not None:
            cmd += ["--max-images", str(profile["max_images"])]
        if profile.get("ops"):
            cmd += ["--ops", ",".join(profile["ops"])]
        cmd += cache_args
        commands.append({"name": "editing", "command": cmd})
    elif p == "baselines":
        cmd = _cmd(sys.executable, "scripts/run_baseline_dataset.py", "--config", profile.get("config", "configs/baseline_protocol.yaml"), "--manifest", profile.get("manifest", "examples/example_dataset/manifest.csv"), "--out", out_dir / profile.get("out_name", "baselines"))
        if profile.get("max_images") is not None:
            cmd += ["--max-images", str(profile["max_images"])]
        if profile.get("run_dummy", True):
            cmd.append("--run-dummy")
        commands.append({"name": "baselines", "command": cmd})
    elif p in {"debug", "all"}:
        configs = profile.get("configs", {})
        if profile.get("run_main_comparison"):
            add_main({"config": configs.get("comparison", "configs/benchmark_baseline.yaml"), "manifest": "examples/example_dataset/manifest.csv", "benchmark_manifest": profile.get("benchmark_manifest"), "max_images": max_images, "run_independent_gaussian": True, "frontend_cache": profile.get("frontend_cache"), "use_frontend_cache": profile.get("use_frontend_cache"), "frontend_cache_manifest": profile.get("frontend_cache_manifest")}, "main_comparison")
        if profile.get("run_ablation"):
            commands.append({"name": "ablation", "command": _cmd(sys.executable, "scripts/run_ablation.py", "--config", configs.get("ablation", "configs/minimal.yaml"), "--ablations", profile.get("ablations", "configs/ablations.yaml"), "--input", "examples/test_image.png", "--out", out_dir / "ablation", "--max-experiments", "2" if p == "debug" else "", "--no-run-metadata")})
        if profile.get("run_stress"):
            stress_manifest = profile.get("benchmark_manifest") or "examples/stress_dataset/manifest.csv"
            commands.append({"name": "stress", "command": _cmd(sys.executable, "scripts/run_stress_benchmark.py", "--config", configs.get("stress", "configs/stress_benchmark.yaml"), "--manifest", stress_manifest, "--out", out_dir / "stress", "--max-images", "4" if p == "debug" else str(max_images or 12), "--no-run-metadata") + cache_args})
        if profile.get("run_editing"):
            edit_manifest = profile.get("benchmark_manifest") or "examples/example_dataset/manifest.csv"
            commands.append({"name": "editing", "command": _cmd(sys.executable, "scripts/run_edit_dataset.py", "--config", configs.get("editing", "configs/editing.yaml"), "--manifest", edit_manifest, "--out", out_dir / "editing", "--max-images", str(max_images or 3), "--max-objects", "1" if p == "debug" else "2", "--ops", "remove,translate", "--no-run-metadata") + cache_args})
        if profile.get("run_baselines"):
            commands.append({"name": "baselines", "command": _cmd(sys.executable, "scripts/run_baseline_dataset.py", "--config", configs.get("baseline", "configs/baseline_protocol.yaml"), "--manifest", "examples/example_dataset/manifest.csv", "--out", out_dir / "baselines", "--max-images", str(max_images or 3), "--run-dummy")})
    else:
        raise ValueError(f"unknown paper profile: {p}")
    return commands


def run_paper_profile(profile: dict, out_dir: str | Path, dry_run: bool = False, stop_on_failure: bool = True) -> dict:
    """运行 paper profile 中的子实验。

    该函数通过 subprocess 调用已有 CLI，避免复制 comparison / stress /
    editing 等实验逻辑，从而保持 paper layer 只是编排层。
    """
    out = Path(out_dir)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    commands = _profile_commands(profile, out)
    save_json({"profile": profile, "commands": commands}, out / "paper_plan.json")
    rows = []
    for step in commands:
        start = datetime.now()
        row = {"name": step["name"], "command": step["command"], "start_time": start.isoformat()}
        if dry_run:
            row.update({"status": "dry_run", "return_code": None})
        else:
            result = subprocess.run(step["command"], cwd=Path.cwd(), text=True, capture_output=True)
            (logs / f"{step['name']}_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
            (logs / f"{step['name']}_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
            row.update({"status": "success" if result.returncode == 0 else "failed", "return_code": result.returncode})
            if result.returncode != 0 and stop_on_failure:
                rows.append(row)
                break
        row["end_time"] = datetime.now().isoformat()
        rows.append(row)
    summary = {"profile": profile.get("profile"), "status": "dry_run" if dry_run else ("success" if all(r["status"] in {"success", "dry_run"} for r in rows) else "failed"), "steps": rows}
    save_json(rows, out / "command_log.json")
    save_json(summary, out / "paper_run_summary.json")
    return summary


def collect_paper_outputs(out_dir: str | Path) -> dict:
    """收集 paper 输出中的各类 summary，供报告和表格导出使用。"""
    out = Path(out_dir)
    candidates = {
        "main_comparison": out / "main_comparison" / "per_method_summary.json",
        "ablation": out / "ablation" / "ablation_summary.json",
        "stress": out / "stress" / "stress_subset_summary.json",
        "editing": out / "editing" / "edit_summary.json",
        "baselines": out / "baselines" / "baseline_summary.json",
        "final_ours": out / "final_ours_debug" / "ours_summary.json",
        "report": out / "reports" / "report.md",
    }
    return {k: v for k, v in candidates.items() if v.exists()}


def generate_paper_tables(out_dir: str | Path, columns_config_path: str | Path | None = None) -> dict:
    """把各子实验 summary 导出成论文表格草稿。"""
    out = Path(out_dir)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    columns_cfg = {}
    if columns_config_path and Path(columns_config_path).exists():
        with open(columns_config_path, "r", encoding="utf-8") as f:
            columns_cfg = yaml.safe_load(f) or {}
    found = collect_paper_outputs(out)
    mapping = {"main_comparison": "main_comparison", "ablation": "ablation", "stress": "stress", "editing": "editing"}
    written = {}
    for key, kind in mapping.items():
        if key not in found:
            continue
        rows = flatten_summary(load_json(found[key]))
        export_paper_table(rows, kind, tables / f"{kind}.csv", tables / f"{kind}.tex", f"{kind} summary.", f"tab:{kind}", columns_cfg)
        written[kind] = {"csv": str(tables / f"{kind}.csv"), "tex": str(tables / f"{kind}.tex")}
    save_json(written, out / "paper_tables_manifest.json")
    return written


def generate_paper_report(out_dir: str | Path) -> Path:
    """生成简短 paper report，列出子实验输出和表格位置。"""
    out = Path(out_dir)
    found = collect_paper_outputs(out)
    lines = ["# ShapeSplat++ Paper Experiment Report", "", "## Outputs"]
    for key, path in found.items():
        lines.append(f"- {key}: `{path}`")
    lines += ["", "## Tables"]
    for path in sorted((out / "tables").glob("*.*")) if (out / "tables").exists() else []:
        lines.append(f"- `{path.relative_to(out)}`")
    lines += ["", "## Notes", "- Debug profiles are smoke tests, not paper results.", "- Optional geometry metrics are only available when point clouds exist.", "- External baseline templates are not real integrations yet."]
    report = out / "paper_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
