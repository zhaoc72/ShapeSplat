from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from shapesplat.experiments.final_readiness import check_final_paper_ready, save_final_readiness_report
from shapesplat.reporting.final_report import generate_final_report
from shapesplat.reporting.final_tables import export_final_tables
from shapesplat.reporting.io import save_json


def load_final_profile(path: str | Path) -> dict:
    """读取 final paper profile。final runner 是 orchestration，不是新算法。"""
    with open(path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f) or {}
    if "profile" not in profile:
        raise ValueError(f"final profile missing profile: {path}")
    return profile


def _cmd(*parts) -> list[str]:
    return [str(p) for p in parts if p is not None and str(p) != ""]


def _write_yaml(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def _resolved_outputs(profile: dict, out: Path) -> dict:
    baselines_dir = out / profile.get("baselines", {}).get("output_dir", "baselines")
    ours_dir = out / profile.get("ours", {}).get("output_dir", "ours")
    return {
        "method_outputs": {
            "ours_full": str(ours_dir),
            "independent_gaussian": str(baselines_dir),
            "identity_mask": str(baselines_dir),
            "scene_union": str(baselines_dir),
        }
    }


def _profile_commands(profile: dict, out: Path) -> list[dict]:
    bench = profile.get("benchmark", {})
    manifest = bench.get("manifest", "data/example_benchmark_v2/manifest.csv")
    max_images = bench.get("max_images")
    commands: list[dict] = []
    runtime_cli = [str(x) for x in profile.get("runtime_cli_args", []) or []]
    if not runtime_cli and profile.get("runtime"):
        # 中文注释：profile 可声明 runtime；CLI 覆盖存在时优先生效。
        rcfg = profile.get("runtime", {})
        if rcfg.get("device"):
            runtime_cli += ["--device", str(rcfg["device"])]
        if rcfg.get("cuda_device") is not None:
            runtime_cli += ["--cuda-device", str(rcfg["cuda_device"])]
        if rcfg.get("require_cuda_for_experiments"):
            runtime_cli.append("--require-cuda")
        if rcfg.get("allow_cpu_fallback"):
            runtime_cli.append("--allow-cpu-fallback")
        if rcfg.get("mixed_precision"):
            runtime_cli.append("--mixed-precision")

    # final_debug 常从 example_dataset 转 benchmark v2；真实 profile 可直接提供 manifest。
    if not Path(manifest).exists():
        commands.append({"name": "create_example_dataset", "command": _cmd(sys.executable, "scripts/create_example_dataset.py", "--out", "examples/example_dataset", "--num-images", "4", "--size", "128")})
        commands.append({"name": "build_example_benchmark", "command": _cmd(sys.executable, "scripts/convert_benchmark.py", "--converter", "generic_folder", "--src", "examples/example_dataset", "--out", Path(manifest).parent, "--source-dataset", "example", "--overwrite")})
    commands.append({"name": "validate_benchmark", "command": _cmd(sys.executable, "scripts/validate_benchmark_v2.py", "--manifest", manifest, "--config", bench.get("config", "configs/final_benchmark.yaml"), "--out", out / "benchmark_validation")})

    ours = profile.get("ours", {})
    if ours.get("run_full", True):
        cmd = _cmd(sys.executable, "scripts/run_ours_benchmark.py", "--config", ours.get("config", "configs/final_ours.yaml"), "--manifest", manifest, "--out", out / ours.get("output_dir", "ours"), "--no-run-metadata")
        if max_images is not None:
            cmd += ["--max-images", str(max_images)]
        if ours.get("save_checkpoint"):
            cmd.append("--save-checkpoint")
        cmd += runtime_cli
        commands.append({"name": "ours_full", "command": cmd})

    variants = profile.get("variants", {})
    if variants.get("enabled", False):
        cmd = _cmd(sys.executable, "scripts/run_ours_variants.py", "--config", variants.get("config", "configs/final_ours.yaml"), "--variants", variants.get("variants", "configs/ours_variants.yaml"), "--manifest", manifest, "--out", out / variants.get("output_dir", "variants"))
        for name in variants.get("variant_names", []) or []:
            cmd += ["--variant", name]
        if variants.get("max_images") is not None:
            cmd += ["--max-images", str(variants["max_images"])]
        cmd += runtime_cli
        commands.append({"name": "ours_variants", "command": cmd})

    baselines = profile.get("baselines", {})
    if baselines.get("enabled", False):
        cmd = _cmd(sys.executable, "scripts/run_comparison.py", "--config", profile.get("comparison", {}).get("config", "configs/benchmark_baseline.yaml"), "--manifest", manifest, "--out", out / baselines.get("output_dir", "baselines"), "--no-ours", "--no-run-metadata")
        if baselines.get("run_independent_gaussian", True):
            cmd.append("--run-independent-gaussian")
        if not baselines.get("run_dummy_baselines", True):
            cmd.append("--no-dummy-baselines")
        if max_images is not None:
            cmd += ["--max-images", str(max_images)]
        commands.append({"name": "baselines", "command": cmd})

    outputs_cfg = out / "final_method_outputs_resolved.yaml"
    commands.append({"name": "write_outputs_config", "internal": "write_outputs_config", "path": str(outputs_cfg)})
    comparison = profile.get("comparison", {})
    if comparison.get("enabled", True):
        cmd = _cmd(sys.executable, "scripts/run_final_comparison.py", "--manifest", manifest, "--methods", comparison.get("method_catalog", "configs/method_catalog.yaml"), "--outputs-config", outputs_cfg, "--config", comparison.get("config", "configs/final_ours.yaml"), "--out", out / comparison.get("output_dir", "comparison"))
        if max_images is not None:
            cmd += ["--max-images", str(max_images)]
        commands.append({"name": "final_comparison", "command": cmd})

    stress = profile.get("stress", {})
    if stress.get("enabled", False):
        if stress.get("create_if_missing"):
            commands.append({"name": "create_stress_dataset", "command": _cmd(sys.executable, "scripts/create_stress_dataset.py", "--out", Path(stress.get("manifest", "examples/stress_dataset/manifest.csv")).parent, "--num-per-subset", stress.get("num_per_subset", 1), "--size", "128")})
        cmd = _cmd(sys.executable, "scripts/run_stress_benchmark.py", "--config", stress.get("config", "configs/stress_benchmark.yaml"), "--manifest", stress.get("manifest", "examples/stress_dataset/manifest.csv"), "--out", out / stress.get("output_dir", "stress"), "--no-run-metadata")
        if stress.get("max_images") is not None:
            cmd += ["--max-images", str(stress["max_images"])]
        commands.append({"name": "stress", "command": cmd})

    editing = profile.get("editing", {})
    if editing.get("enabled", False):
        if editing.get("create_if_missing"):
            commands.append({"name": "create_example_dataset", "command": _cmd(sys.executable, "scripts/create_example_dataset.py", "--out", Path(editing.get("manifest", "examples/example_dataset/manifest.csv")).parent, "--num-images", "4", "--size", "128")})
        cmd = _cmd(sys.executable, "scripts/run_edit_dataset.py", "--config", editing.get("config", "configs/editing.yaml"), "--manifest", editing.get("manifest", "examples/example_dataset/manifest.csv"), "--out", out / editing.get("output_dir", "editing"), "--max-objects", editing.get("max_objects", 2), "--no-run-metadata")
        if editing.get("max_images") is not None:
            cmd += ["--max-images", str(editing["max_images"])]
        commands.append({"name": "editing", "command": cmd})
    return commands


def collect_final_outputs(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    candidates = {
        "ours": out / "ours" / "ours_summary.json",
        "variants": out / "variants" / "variant_summary.json",
        "comparison": out / "comparison" / "final_method_summary.json",
        "stress": out / "stress" / "stress_subset_summary.json",
        "editing": out / "editing" / "edit_summary.json",
        "report": out / "report" / "final_report.md",
    }
    found = {k: str(p) for k, p in candidates.items() if p.exists()}
    if (out / "tables").exists():
        found["tables"] = [str(p) for p in sorted((out / "tables").glob("*.*"))]
    return found


def write_final_run_summary(summary: dict, out_dir: str | Path) -> None:
    save_json(summary, Path(out_dir) / "final_run_summary.json")


def run_final_paper_experiment(profile: dict, out_dir: str | Path, dry_run: bool = False, stop_on_failure: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_yaml(profile, out / "final_profile_resolved.yaml")
    readiness = check_final_paper_ready(profile, out, strict=bool(profile.get("readiness", {}).get("strict", False)))
    save_final_readiness_report(readiness, out / "readiness")
    commands = _profile_commands(profile, out)
    save_json(commands, out / "final_plan.json")
    rows: list[dict] = []
    for step in commands:
        start = time.time()
        row = {"name": step["name"], "status": "pending", "command": step.get("command"), "output_dir": step.get("output_dir")}
        try:
            if dry_run:
                row["status"] = "dry_run"
            elif step.get("internal") == "write_outputs_config":
                _write_yaml(_resolved_outputs(profile, out), Path(step["path"]))
                row["status"] = "success"
            else:
                result = subprocess.run(step["command"], cwd=Path.cwd(), text=True, capture_output=True)
                logs = out / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                (logs / f"{step['name']}_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
                (logs / f"{step['name']}_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
                row["return_code"] = result.returncode
                row["status"] = "success" if result.returncode == 0 else "failed"
                if result.returncode != 0:
                    row["error"] = result.stderr[-2000:]
                    rows.append(row)
                    if stop_on_failure:
                        break
                    continue
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            rows.append(row)
            if stop_on_failure:
                break
            continue
        finally:
            row["duration_sec"] = time.time() - start
        rows.append(row)
    status = "dry_run" if dry_run else ("success" if all(r["status"] in {"success", "dry_run"} for r in rows) else "failed")
    summary = {"profile": profile.get("profile"), "status": status, "steps": rows, "outputs": collect_final_outputs(out)}
    save_json(rows, out / "command_log.json")
    save_json({"status": status, "readiness_ready": readiness.get("ready"), "outputs_found": summary["outputs"]}, out / "final_artifact_check.json")
    write_final_run_summary(summary, out)
    return summary
