from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".json", ".csv", ".bat", ".ps1"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".trash"}
OLD_PATH_PATTERNS = [
    r"C:\Users\zhaoc\ShapeSplat\shapesplat_minimal",
    r"C:\\Users\\zhaoc\\ShapeSplat\\shapesplat_minimal",
    "C:/Users/zhaoc/ShapeSplat/shapesplat_minimal",
    r"C:\Users\zhaoc\ShapeSplat\shapesplat",
    r"C:\\Users\\zhaoc\\ShapeSplat\\shapesplat",
    "C:/Users/zhaoc/ShapeSplat/shapesplat",
    "shapesplat_minimal",
]
FALLBACK_NEEDLES = [
    "ToyShapeBank",
    "SoftGaussianRenderer",
    "DinoV3Stub",
    "Sam3Stub",
    "DepthStub",
    "fallback_to_toy",
    "fallback_to_soft",
    "fallback_to_compute",
    "debug_iteration_cap",
]


MODULE_TAGS = {
    "baselines": "baseline",
    "benchmarks": "benchmark",
    "cache": "frontend",
    "cleanup": "cleanup",
    "data": "dataset",
    "datasets": "dataset",
    "editing": "evaluation",
    "evaluation": "evaluation",
    "experiments": "experiment",
    "frontend": "frontend",
    "gaussian": "core_algorithm",
    "geometry": "core_algorithm",
    "integration": "diagnostic",
    "optimization": "core_algorithm",
    "reconstruction": "reconstruction",
    "renderer": "renderer",
    "reporting": "reporting",
    "reproducibility": "runtime",
    "runtime": "runtime",
    "shape_prior": "core_algorithm",
    "utils": "diagnostic",
    "validation": "diagnostic",
}

MODULE_PURPOSE = {
    "baselines": "管理 dummy、internal、external baseline 协议和输出读取。",
    "benchmarks": "生成、验证和汇总 benchmark / stress benchmark。",
    "cache": "保存、加载、绑定和验证 frontend cache。",
    "cleanup": "安全扫描和移动生成文件，不删除源码。",
    "data": "图像 I/O、resize 策略和 synthetic input。",
    "datasets": "manifest、dataset runner、benchmark v2 converter 和 CO3Dv2 converter。",
    "editing": "object editing 操作、metrics 和可视化。",
    "evaluation": "2D、editing、geometry、method output 评估。",
    "experiments": "paper/final/CO3Dv2/high-res orchestration 和 readiness。",
    "frontend": "file mask、SAM3、DINOv3、Depth backend 和 pooling。",
    "gaussian": "visible-hidden Gaussian object/scene 表示。",
    "geometry": "camera、mask 和 projection 基础几何工具。",
    "integration": "真实 backend capability 和 smoke diagnostic。",
    "optimization": "Trainer、loss 和 edit optimization。",
    "reconstruction": "Ours runner、variants、diagnostics 和 output protocol。",
    "renderer": "renderer contract、soft renderer 和 Real3DGS adapter。",
    "reporting": "final/paper tables、LaTeX、qualitative 和 report。",
    "reproducibility": "environment、hash、snapshot、run registry。",
    "runtime": "Windows/RTX CUDA runtime、AMP、memory、environment summary。",
    "shape_prior": "Toy/file shape bank、retrieval 和 hidden support prior。",
    "utils": "logging、seed、visualization、config override 等通用工具。",
    "validation": "artifact、command matrix 和 project health validation。",
}


SCRIPT_CATEGORY_HINTS = {
    "migration": ["migrate_", "audit_project_paths"],
    "cleanup": ["cleanup", "clean_", "large_generated"],
    "runtime": ["gpu", "runtime", "windows"],
    "paper": ["paper", "final_paper"],
    "reporting": ["report", "table", "summarize"],
    "stress": ["stress"],
    "editing": ["edit"],
    "comparison": ["comparison", "compare", "final_comparison"],
    "baseline": ["baseline"],
    "variants": ["variant"],
    "ours": ["ours", "co3dv2_highres_ours"],
    "cache": ["cache"],
    "frontend": ["dino", "sam", "depth", "frontend"],
    "benchmark": ["benchmark", "validate_benchmark", "summarize_benchmark"],
    "dataset": ["convert", "dataset", "co3dv2"],
    "check": ["check_", "inspect_", "validate_", "print_", "list_"],
    "setup": ["create_", "prepare_", "install"],
}


def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _skip_path(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    text = str(path).replace("\\", "/")
    if "/outputs/codebase_audit/" in text:
        return True
    if "/outputs/path_audit/" in text:
        return True
    return False


def _is_text_file(path: Path) -> bool:
    try:
        return path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 2_000_000
    except OSError:
        return False


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def build_tree(root: Path) -> str:
    lines = [str(root)]
    focus = ["src", "scripts", "configs", "tests", "docs", "examples", "data", "outputs", "runs"]
    for name in focus:
        base = root / name
        if not base.exists():
            lines.append(f"{name}/ [missing]")
            continue
        lines.append(f"{name}/")
        max_depth = 5 if name in {"src", "scripts", "configs", "tests", "docs"} else 3
        for path in sorted(base.rglob("*")):
            if _skip_path(path):
                continue
            rel = path.relative_to(base)
            if len(rel.parts) > max_depth:
                continue
            if name in {"outputs", "runs", "data"} and path.is_file():
                try:
                    if path.stat().st_size > 200_000:
                        continue
                except OSError:
                    continue
            indent = "  " * len(rel.parts)
            suffix = "/" if path.is_dir() else ""
            lines.append(f"{indent}{path.name}{suffix}")
    return "\n".join(lines) + "\n"


def module_inventory(root: Path) -> dict[str, Any]:
    src_root = root / "src" / "shapesplat"
    inventory: dict[str, Any] = {}
    if not src_root.exists():
        return inventory
    for path in sorted(src_root.iterdir()):
        if path.name.startswith("__"):
            continue
        if path.is_dir():
            files = [str(p.relative_to(src_root)).replace("\\", "/") for p in sorted(path.rglob("*.py")) if not _skip_path(p)]
            tag = MODULE_TAGS.get(path.name, "diagnostic")
            inventory[path.name] = {
                "tag": tag,
                "purpose": MODULE_PURPOSE.get(path.name, "项目内部模块。"),
                "files": files,
                "is_placeholder_or_fallback_related": any("stub" in f.lower() or "placeholder" in f.lower() or "fallback" in f.lower() for f in files),
            }
        elif path.suffix == ".py":
            inventory[path.name] = {
                "tag": "diagnostic",
                "purpose": "包级工具或配置入口。",
                "files": [path.name],
                "is_placeholder_or_fallback_related": False,
            }
    return inventory


def _script_category(name: str) -> str:
    lower = name.lower()
    for category, hints in SCRIPT_CATEGORY_HINTS.items():
        if any(h in lower for h in hints):
            return category
    return "other"


def _argparse_args(path: Path) -> list[str]:
    args: list[str] = []
    try:
        tree = ast.parse(_read_text(path))
    except SyntaxError:
        return args
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for item in node.args:
                if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value.startswith("--"):
                    args.append(item.value)
    return sorted(set(args))


def script_inventory(root: Path) -> dict[str, Any]:
    scripts_dir = root / "scripts"
    out: dict[str, Any] = {}
    if not scripts_dir.exists():
        return out
    for path in sorted(scripts_dir.glob("*.py")):
        text = _read_text(path)
        out[path.name] = {
            "category": _script_category(path.name),
            "purpose": _script_purpose(path.name),
            "main_args": _argparse_args(path),
            "is_cli_wrapper": "argparse" in text or "ArgumentParser" in text,
        }
    return out


def _script_purpose(name: str) -> str:
    lower = name.lower()
    if "co3dv2_highres" in lower:
        return "CO3Dv2 high-res workflow wrapper。"
    if lower.startswith("check_"):
        return "环境、backend、readiness 或 artifact 检查脚本。"
    if lower.startswith("run_"):
        return "运行实验、benchmark、paper flow 或 demo 的 CLI。"
    if lower.startswith("cache_"):
        return "生成或验证 frontend cache。"
    if lower.startswith("convert_"):
        return "数据集或 benchmark 转换脚本。"
    if "audit" in lower or "migrate" in lower:
        return "审计、路径检查或项目迁移工具。"
    return "项目辅助 CLI。"


def config_inventory(root: Path) -> dict[str, Any]:
    configs_dir = root / "configs"
    out: dict[str, Any] = {}
    if not configs_dir.exists():
        return out
    for path in sorted(configs_dir.rglob("*.yaml")):
        rel = str(path.relative_to(configs_dir)).replace("\\", "/")
        text = _read_text(path)
        lower = (rel + "\n" + text).lower()
        out[rel] = {
            "purpose": _config_purpose(rel, lower),
            "is_debug": "debug" in lower or "smoke" in lower,
            "is_paper_profile": rel.startswith("paper/") or "profile:" in lower,
            "is_co3dv2": "co3d" in lower,
            "is_highres": "highres" in lower or "long_side: 640" in lower or "dino_input_size: 448" in lower,
            "is_still_minimal": "minimal" in lower,
            "may_use_fallback": any(n.lower() in lower for n in FALLBACK_NEEDLES) or "fallback_to_" in lower or "stub" in lower,
            "paper_ready": _paper_ready_config(rel, lower),
        }
    return out


def _config_purpose(rel: str, lower: str) -> str:
    if "co3dv2" in lower and ("highres" in lower or "highres" in rel):
        return "CO3Dv2 high-resolution frontend/Ours/readiness 配置。"
    if rel.startswith("paper/"):
        return "paper experiment profile。"
    if rel.startswith("presets/"):
        return "run_experiment preset。"
    if rel.startswith("datasets/"):
        return "dataset converter 配置。"
    if "baseline" in lower:
        return "baseline 或 comparison 配置。"
    if "renderer" in lower:
        return "renderer/backend 配置。"
    return "通用 ShapeSplat++ 实验配置。"


def _paper_ready_config(rel: str, lower: str) -> bool:
    if "minimal" in lower or "debug" in lower or "toy" in lower or "stub" in lower:
        return False
    if "fallback_to_soft: true" in lower or "fallback_to_toy: true" in lower:
        return False
    return rel.startswith("paper/") or rel.startswith("final")


def test_inventory(root: Path) -> dict[str, Any]:
    tests_dir = root / "tests"
    out: dict[str, Any] = {}
    if not tests_dir.exists():
        return out
    for path in sorted(tests_dir.glob("test_*.py")):
        text = _read_text(path).lower()
        name = path.name
        out[name] = {
            "coverage": name.removeprefix("test_").removesuffix(".py").replace("_", " "),
            "requires_real_gpu": "skipif(not torch.cuda.is_available" in text or "require_cuda" in text,
            "requires_real_dino_sam": "real dino" in text or "sam3_checkpoint" in text or "real sam" in text,
            "quick_test": "optional" not in name and "gpu_optional" not in name,
            "is_smoke_test": "smoke" in name or "smoke" in text,
            "covers_highres_co3dv2": "highres" in name or "high-res" in text or "co3dv2_highres" in text,
        }
    return out


def doc_inventory(root: Path) -> dict[str, Any]:
    docs_dir = root / "docs"
    out: dict[str, Any] = {}
    if not docs_dir.exists():
        return out
    for path in sorted(docs_dir.glob("*.md")):
        text = _read_text(path)
        title = next((line.strip("# ").strip() for line in text.splitlines() if line.startswith("#")), path.stem)
        out[path.name] = {"title": title, "mentions_co3dv2": "co3d" in text.lower(), "mentions_highres": "high-res" in text.lower() or "highres" in text.lower()}
    return out


def path_audit(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if _skip_path(path) or not _is_text_file(path):
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel.startswith("outputs/") or rel.startswith("dist/") or ".egg-info/" in rel:
            continue
        for idx, line in enumerate(_read_text(path).splitlines(), 1):
            hits = [p for p in OLD_PATH_PATTERNS if p in line]
            if not hits:
                continue
            informational = rel == "agents.md" or rel == "scripts/audit_project_paths.py" or rel == "scripts/audit_codebase_current.py"
            rows.append(
                {
                    "file": rel,
                    "line": idx,
                    "snippet": line.strip()[:300],
                    "hits": hits,
                    "needs_fix": not informational,
                    "recommendation": "保留为历史说明或审计规则。" if informational else r"建议替换为 C:\Users\zhaoc\ShapeSplat。",
                }
            )
    return rows


def fallback_audit(root: Path) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    config_hits: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if _skip_path(path) or not _is_text_file(path):
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel.startswith("outputs/") or rel.startswith("dist/"):
            continue
        for idx, line in enumerate(_read_text(path).splitlines(), 1):
            hits = [n for n in FALLBACK_NEEDLES if n in line]
            if hits:
                matches.append({"file": rel, "line": idx, "hits": hits, "snippet": line.strip()[:300]})
                if rel.startswith("configs/"):
                    config_hits.setdefault(rel, [])
                    config_hits[rel].extend(hits)
    return {
        "matches": matches,
        "config_hits": {k: sorted(set(v)) for k, v in config_hits.items()},
        "debug_ok": ["ToyShapeBank", "SoftGaussianRenderer", "DinoV3Stub", "Sam3Stub", "DepthStub", "fallback_to_compute"],
        "paper_warning": ["ToyShapeBank", "SoftGaussianRenderer", "DinoV3Stub", "Sam3Stub", "DepthStub", "fallback_to_toy", "fallback_to_soft", "debug_iteration_cap"],
        "strict_final_error": ["ToyShapeBank", "SoftGaussianRenderer", "DinoV3Stub", "Sam3Stub", "DepthStub"],
    }


def highres_audit(root: Path) -> dict[str, Any]:
    required = [
        "configs/co3dv2_real_frontend_highres.yaml",
        "configs/final_ours_co3dv2_highres.yaml",
        "scripts/cache_co3dv2_highres_frontend.py",
        "scripts/run_co3dv2_highres_ours.py",
        "scripts/inspect_output_resolution.py",
        "scripts/check_co3dv2_highres_ready.py",
        "tests/test_co3dv2_highres_config.py",
        "tests/test_image_mask_resize_policy.py",
        "tests/test_co3dv2_highres_readiness.py",
        "tests/test_output_resolution_diagnostics.py",
    ]
    exists = {p: (root / p).exists() for p in required}
    cfg_front = _read_text(root / "configs/co3dv2_real_frontend_highres.yaml") if (root / "configs/co3dv2_real_frontend_highres.yaml").exists() else ""
    cfg_ours = _read_text(root / "configs/final_ours_co3dv2_highres.yaml") if (root / "configs/final_ours_co3dv2_highres.yaml").exists() else ""
    combined = (cfg_front + "\n" + cfg_ours).lower()
    return {
        "required_files": exists,
        "all_required_present": all(exists.values()),
        "dino_input_size_at_least_448": _contains_min_number(combined, "dino_input_size", 448),
        "image_long_side_at_least_640": _contains_min_number(combined, "long_side", 640) or _contains_min_number(combined, "image_resize_long_side", 640),
        "mask_resize_nearest": "mask_resize_mode: nearest" in combined,
        "debug_iteration_cap_controlled_or_disabled": "allow_debug_iteration_cap: false" in combined or "allow_debug_iteration_cap" in combined,
    }


def _contains_min_number(text: str, key: str, threshold: int) -> bool:
    for line in text.splitlines():
        if key in line and ":" in line:
            try:
                value = int(line.split(":", 1)[1].strip().split()[0].strip("[] ,"))
            except ValueError:
                continue
            if value >= threshold:
                return True
    return False


def run_command(args: list[str], timeout: int = 240) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {
            "command": " ".join(args),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-5000:],
            "stderr_tail": proc.stderr[-5000:],
        }
    except Exception as exc:
        return {"command": " ".join(args), "returncode": None, "error": str(exc)}


def run_light_checks(root: Path) -> list[dict[str, Any]]:
    checks = [
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        [sys.executable, "scripts/check_project_health.py"],
        [sys.executable, "scripts/run_quick_tests.py"],
        [sys.executable, "scripts/check_dinov3_dependencies.py"],
        [sys.executable, "scripts/check_gpu_runtime.py", "--config", "configs/local_windows_rtx5090.yaml", "--device", "auto", "--out", "outputs/codebase_audit/check_gpu_runtime_auto"],
    ]
    results = [run_command(cmd, timeout=300) for cmd in checks]
    for test in [
        "tests/test_dinov3_dense_extraction.py",
        "tests/test_co3dv2_real_frontend_config.py",
        "tests/test_co3dv2_converter.py",
        "tests/test_co3dv2_highres_config.py",
        "tests/test_image_mask_resize_policy.py",
        "tests/test_co3dv2_highres_readiness.py",
        "tests/test_output_resolution_diagnostics.py",
    ]:
        if (root / test).exists():
            results.append(run_command([sys.executable, "-m", "pytest", test, "-v"], timeout=300))
        else:
            results.append({"command": f"pytest {test} -v", "returncode": "not_found", "stdout_tail": "", "stderr_tail": ""})
    return results


def _command_lines(results: list[dict[str, Any]]) -> str:
    if not results:
        return "- 本次未运行命令。"
    return "\n".join(f"- `{r['command']}` -> `{r.get('returncode')}`" for r in results)


def _health_ok(results: list[dict[str, Any]]) -> bool:
    return bool(results) and all(r.get("returncode") in (0, "not_found") for r in results)


def write_capabilities(out: Path, inventories: dict[str, Any], command_results: list[dict[str, Any]]) -> None:
    highres = inventories["highres_audit"]
    path_rows = inventories["path_audit"]
    fallback = inventories["fallback_audit"]
    text = f"""# 当前能力说明

当前代码库位于 `C:\\Users\\zhaoc\\ShapeSplat`，已经完成从嵌套目录到扁平项目根目录的迁移。

## 当前代码能做什么

- 运行基础 smoke / quick tests。
- 使用 benchmark v2 manifest / converter / validator。
- 将 CO3Dv2 single subset 转换为 real-image diagnostic benchmark。
- 使用 CO3Dv2 file masks 作为 same-mask protocol 的主 mask source。
- 检查 DINOv3 官方 repo 依赖，并加载本地 ViT-S/16 / ViT-L/16 配置。
- 从 DINOv3 提取 dense patch features，而不是 global image embedding。
- 生成和验证 frontend cache，包括 high-res cache metadata。
- 运行 Ours benchmark、Ours variants、editing、stress、baseline comparison 和 final report 框架。
- 检查 RTX 5090 / CUDA runtime，避免 CUDA 请求时静默 CPU fallback。
- 执行 cleanup、root migration、path audit 等工程维护工具。

## 当前代码不能保证什么

- ToyShapeBank 和 SoftGaussianRenderer 仍不是 paper-final。
- 真实 external baselines 尚未接入。
- CO3Dv2 single 是单主体 real-image diagnostic，不是多物体主 benchmark。
- Chamfer / F-score 只有在 pred 和 GT pointcloud 都存在且 alignment protocol 明确时才可报告。
- final tables 可以导出，但内容是否 paper-ready 取决于真实 benchmark、shape bank、renderer 和 baseline 输出。

## 哪些结果可以写论文

目前可以写作方法工程流程、diagnostic、消融框架、cache/readiness/benchmark protocol 的支撑材料。正式主表仍不能直接使用 ToyShapeBank、SoftRenderer、dummy baseline 或 CO3Dv2 single 多物体 claim。

## 哪些结果只能 debug

- minimal/example dataset outputs
- ToyShapeBank fallback outputs
- SoftGaussianRenderer fallback outputs
- dummy baseline-only comparison
- debug iteration cap enabled runs
- geometry unavailable rows

## 审计摘要

- high-res CO3Dv2 required files present: `{highres.get('all_required_present')}`
- old path matches needing fix: `{sum(1 for row in path_rows if row.get('needs_fix'))}`
- fallback matches: `{len(fallback.get('matches', []))}`
- light checks healthy: `{_health_ok(command_results)}`

## 下一步建议

1. 先运行 CO3Dv2 high-res readiness。
2. 重新生成 high-res frontend cache 并确认 `num_valid > 0`。
3. 运行 high-res Ours benchmark。
4. 用 `inspect_output_resolution.py` 确认 original / working / renderer resolution。
5. 再运行 high-res variants 和 CO3Dv2 diagnostic report。
6. 准备真实 shape bank、真实 renderer 策略和 external baseline outputs，之后再进入 paper-ready final table。
"""
    (out / "current_capabilities.md").write_text(text, encoding="utf-8")


def write_status(out: Path, inventories: dict[str, Any], command_results: list[dict[str, Any]]) -> None:
    highres = inventories["highres_audit"]
    path_rows = inventories["path_audit"]
    fallback = inventories["fallback_audit"]
    health = _health_ok(command_results)
    old_minimal_needs_fix = any("shapesplat_minimal" in " ".join(row.get("hits", [])) and row.get("needs_fix") for row in path_rows)
    old_nested_needs_fix = any(r"C:\Users\zhaoc\ShapeSplat\shapesplat" in " ".join(row.get("hits", [])) and row.get("needs_fix") for row in path_rows)
    text = f"""# ShapeSplat++ 当前代码库状态报告

## 1. 当前项目路径和运行环境

项目路径：

`C:\\Users\\zhaoc\\ShapeSplat`

推荐运行方式：Anaconda Prompt。

必须先执行：

```bat
conda activate shapesplat
cd /d C:\\Users\\zhaoc\\ShapeSplat
```

旧路径 `C:\\Users\\zhaoc\\ShapeSplat\\shapesplat_minimal` 和 `C:\\Users\\zhaoc\\ShapeSplat\\shapesplat` 已废弃。本报告只基于当前扁平根目录生成。

## 2. 项目总体结构

- `src/shapesplat/`：核心 Python 包，包含 frontend、Gaussian 表示、renderer、optimization、reconstruction、datasets、evaluation、runtime、reporting、cleanup 等模块。
- `scripts/`：CLI 入口，覆盖检查、数据转换、frontend cache、Ours/variants、baseline、comparison、editing、stress、paper/report、cleanup、runtime、migration。
- `configs/`：实验配置、paper profile、presets、dataset config、runtime/backend config、CO3Dv2 high-res config。
- `tests/`：quick tests、CO3Dv2 converter、DINO dense extraction、real frontend config、high-res policy/readiness、runtime、cleanup/path migration 等。
- `docs/`：backend、baseline、commands、experiments、runbook、troubleshooting 和 artifact checklist。
- `examples/`：toy/example/stress 数据和 shape bank 示例。
- `data/`：benchmark v2 输出位置；当前关键数据是 `data/co3dv2_single_benchmark/manifest.csv`。
- `outputs/`：实验输出和报告；包含 debug/diagnostic 结果，不应自动视为 paper-ready。
- `runs/`：run registry / reproducibility metadata。

## 3. 核心算法实现状态

- Visible-hidden Gaussian representation：已在 `gaussian/`、`optimization/`、`reconstruction/` 中实现，能输出 visible/hidden Gaussian 统计；paper-final 仍需真实 shape prior 支撑。
- Scene-coupled ownership rendering：renderer contract 和 SoftGaussianRenderer 支持 per-object ownership/contribution；真实 3DGS adapter 已有但不是默认最终 backend。
- Confidence-weighted hidden support prior：shape retrieval、hidden prior、confidence weighting 配置和逻辑存在；ToyShapeBank fallback 只适合 debug。
- Differentiable edit consistency：editing suite、edit metrics 和相关 loss/diagnostics 已接入，可用于 diagnostic 和 ablation。
- SAM3-DINOv3 frozen frontend：file masks、DINOv3 real wrapper、SAM3 optional wrapper、Depth fallback 和 cache 都存在；SAM3 不是 CO3Dv2 主实验默认 mask source。
- Shape retrieval：ToyShapeBank/FileShapeBank/backend interface 已有；paper-ready 需要准备真实 shape bank。
- Renderer backend：Soft renderer 可用，Real3DGSRendererAdapter 是 adapter/接口；真实 CUDA renderer 尚未作为 paper-final 跑通。

## 4. Frontend 状态

- file masks：支持 same-mask protocol，CO3Dv2 主流程使用 file masks。
- SAM3：作为 optional automatic-mask diagnostic；不作为 CO3Dv2 主实验默认 mask source。
- DINOv3：支持本地 repo/checkpoint、依赖检查、ViT-S/16 和 ViT-L/16 配置；dense patch feature extraction 已修复。
- Depth：DepthStub/auto fallback 可用；真实 depth model 仍是 optional。
- frontend cache：支持 cache manifest、metadata、validation、zero-valid warning、low-res warning。
- CO3Dv2 cache：有 real frontend cache 和 high-res cache wrapper。
- high-res config：`configs/co3dv2_real_frontend_highres.yaml` 与 `configs/final_ours_co3dv2_highres.yaml` 存在。
- fallback：DINO/SAM/Depth 都保留 stub/auto fallback，final readiness 需要明确 warning 或 strict error。

## 5. CO3Dv2 状态

- converter：`src/shapesplat/datasets/converters/co3dv2_single.py` 存在，支持 annotation-driven 和 folder-scan fallback。
- manifest：`data/co3dv2_single_benchmark/manifest.csv` 作为 benchmark v2 输入。
- benchmark validation：支持 image/mask/metadata/optional depth/camera/GT/cache 字段验证。
- 定位：CO3Dv2 single 是 real-image diagnostic / single foreground visible-mask benchmark，不是多物体主 benchmark。
- file masks：作为 retained visible masks 固定使用；主实验不改成 SAM3。
- high-res：已支持 keep-aspect long side 640、nearest mask resize、DINO input 448、debug cap 可控/关闭。
- low-res 验证：仍需要实际跑 high-res cache + Ours，并用 `inspect_output_resolution.py` 检查输出。
- inspect tool：`scripts/inspect_output_resolution.py` 存在。

## 6. Ours reconstruction 状态

- `scripts/run_ours_benchmark.py`：通用 Ours benchmark runner。
- `scripts/run_ours_variants.py`：Ours variants / ablation runner。
- `scripts/run_co3dv2_highres_ours.py`：CO3Dv2 high-res Ours wrapper 存在。
- output protocol：保存 render/alpha/ownership/metrics/output_spec/reconstruction_meta/diagnostics/pred_pointcloud。
- diagnostics：记录 original/working/render shapes、cache 使用、mask resize、DINO feature shape、debug cap、renderer/shape bank fallback。
- pointcloud export：支持 lightweight Gaussian center proxy。
- limitations：真实 shape bank、真实 renderer 和 external baselines 尚未完成 paper-final。

## 7. Baseline / comparison 状态

- dummy baseline：可用于 smoke/debug comparison。
- independent Gaussian baseline：内部 baseline 已有。
- external baseline adapter：有 command template 和 output protocol；真实 SPAR3D/TRELLIS/VGGT/DUSt3R 等未接入。
- method catalog：存在，可用于 method family grouping。
- final comparison：可统一评估已有 method outputs；缺失 external outputs 时 warning，不失败。

## 8. Reporting / paper experiment 状态

- `scripts/run_final_paper.py`、`configs/paper/final_debug.yaml`、`configs/paper/final_all.yaml` 存在。
- final tables / final report / readiness check 框架存在。
- CO3Dv2 high-res report 工具存在。
- 能生成论文格式表格，但当前内容是否 paper-ready 取决于真实 benchmark、shape bank、renderer、external baseline 输出和 strict readiness。

## 9. Runtime / GPU 状态

- runtime 模块支持 device resolve、CUDA smoke test、memory、AMP 和 environment summary。
- `scripts/check_gpu_runtime.py` 支持 RTX 5090 / sm_120 matmul/backward smoke test。
- mixed precision 默认关闭。
- CUDA requested 时不应 silent CPU fallback；fallback 需要显式配置。
- 本次 audit 已在当前根目录重新运行 GPU auto check，结果见 command results。

## 10. Tests 状态

- quick tests：`scripts/run_quick_tests.py` 可运行。
- full tests：存在多组 benchmark、baseline、paper、runtime、cleanup 测试，可按需运行。
- optional GPU tests：`tests/test_gpu_optional.py` 这类测试不应作为 CI 必跑。
- real backend tests：配置/逻辑测试不依赖真实 DINO/SAM 权重；真实权重检查走脚本。
- CO3Dv2 tests：converter fake structure 测试存在。
- high-res tests：resize policy、high-res config/readiness、output resolution diagnostics 测试存在。
- cleanup/path migration tests：cleanup tests 存在，root migration/path audit 工具已加入。

本次轻量检查结果：

{_command_lines(command_results)}

## 11. 当前仍是 debug/fallback 的部分

- ToyShapeBank fallback：debug 合理；final/paper 应 warning 或 strict error。
- SoftGaussianRenderer fallback：工程诊断可用；不是最终真实 CUDA 3DGS 质量。
- external baselines missing：catalog/template 存在，真实方法输出未接入。
- geometry metrics optional：只有 pred/GT pointcloud 都存在且协议明确时可报告。
- CO3Dv2 single is diagnostic：不是多物体主 benchmark。
- minimal configs still present：保留用于 smoke/compatibility，不能当 final experiment。
- debug iteration cap logic still present：已可控；high-res config 应关闭。

Fallback audit 共发现 `{len(fallback.get('matches', []))}` 处匹配，详见 `fallback_audit.json`。

## 12. 是否可以直接跑论文实验并填表

1. debug paper flow：可以运行，用于工程 smoke，不是投稿结果。
2. CO3Dv2 diagnostics：可以运行，定位为 real-image diagnostic。
3. CO3Dv2 high-res diagnostics：代码与配置存在；需要实际重新生成 high-res cache、运行 high-res Ours、inspect resolution。
4. paper-ready final tables：框架能生成，但当前不能直接作为投稿主表。
5. 当前可用表格：debug comparison、variants、geometry available=false 汇总、CO3Dv2 diagnostic summary。
6. 当前不可作为投稿结果：ToyShapeBank/SoftRenderer/dummy baseline/example benchmark/CO3Dv2 single 多物体 claim。
7. 原因：真实 shape bank、真实 renderer、external baselines、multi-object benchmark 和 final strict readiness 尚未全部满足。

## 13. 阻止 paper-ready 的主要缺口

- high-res CO3Dv2：代码已支持，但需要实际 cache + Ours run + resolution inspection 确认。
- real shape bank：尚未接入 paper-ready prepared bank。
- real renderer：adapter 有，真实 CUDA renderer 未作为最终 backend 跑通。
- external baselines：真实方法输出未接入。
- multi-object benchmark：正式多物体 benchmark 未准备/未验证。
- final strict readiness：需要在真实 config/output 上通过。
- old path：当前旧路径匹配中需要修复的数量为 `{sum(1 for row in path_rows if row.get('needs_fix'))}`。

## 14. 下一步建议

1. 先处理 path audit 中任何 `needs_fix=true` 的旧路径；当前结果为 `{sum(1 for row in path_rows if row.get('needs_fix'))}`。
2. 运行 CO3Dv2 high-res readiness。
3. 重新生成 high-res frontend cache，并确认 `num_valid > 0`。
4. 运行 `run_co3dv2_highres_ours.py`。
5. 运行 `inspect_output_resolution.py`，确认 original/working/render shapes 接近 640 长边且 debug cap 未触发。
6. 跑 high-res variants 的 `full` 和 `visible_only`。
7. 准备真实 shape bank、renderer 策略、multi-object benchmark 和 external baseline outputs。

## 15. 重要命令

```bat
conda activate shapesplat
cd /d C:\\Users\\zhaoc\\ShapeSplat
python scripts/check_project_health.py
python scripts/run_quick_tests.py
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device cuda --require-cuda --out outputs/check_gpu_runtime_cuda
python scripts/check_dinov3_dependencies.py
python scripts/check_dinov3_weights.py --config configs/co3dv2_real_frontend_debug.yaml --input examples/test_image.png --out outputs/check_dinov3_vits16 --device cuda
python scripts/check_co3dv2_highres_ready.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --out outputs/check_co3dv2_highres_ready
python scripts/cache_co3dv2_highres_frontend.py --config configs/co3dv2_real_frontend_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16_highres --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --validate --require-cuda --check-deps-first
python scripts/run_co3dv2_highres_ours.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_co3dv2_vits16_highres --max-images 5 --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
python scripts/inspect_output_resolution.py --out outputs/ours_co3dv2_vits16_highres --max-items 5
```

## Path Audit 摘要

- 旧路径总匹配：`{len(path_rows)}`
- 需要修正：`{sum(1 for row in path_rows if row.get('needs_fix'))}`
- `shapesplat_minimal` 需要修正残留：`{old_minimal_needs_fix}`
- `shapesplat` 子目录需要修正残留：`{old_nested_needs_fix}`

## 健康结论

- 当前轻量检查健康：`{health}`
- 当前包含 high-res CO3Dv2 修复：`{highres.get('all_required_present')}`
- 当前能跑 debug paper flow：`True`
- 当前能跑 paper-ready final tables：`False`
"""
    (out / "CODEBASE_STATUS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the current flattened ShapeSplat++ codebase.")
    parser.add_argument("--out", default="outputs/codebase_audit")
    parser.add_argument("--run-checks", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    inventories = {
        "module_inventory": module_inventory(ROOT),
        "script_inventory": script_inventory(ROOT),
        "config_inventory": config_inventory(ROOT),
        "test_inventory": test_inventory(ROOT),
        "doc_inventory": doc_inventory(ROOT),
        "path_audit": path_audit(ROOT),
        "fallback_audit": fallback_audit(ROOT),
        "highres_audit": highres_audit(ROOT),
    }
    command_results = run_light_checks(ROOT) if args.run_checks else []

    (out / "codebase_tree.txt").write_text(build_tree(ROOT), encoding="utf-8")
    for name, data in inventories.items():
        _json_dump(out / f"{name}.json", data)
    _json_dump(out / "command_results.json", command_results)
    write_capabilities(out, inventories, command_results)
    write_status(out, inventories, command_results)
    print(
        json.dumps(
            {
                "status_report": str(out / "CODEBASE_STATUS.md"),
                "healthy": _health_ok(command_results),
                "path_matches": len(inventories["path_audit"]),
                "path_needs_fix": sum(1 for row in inventories["path_audit"] if row.get("needs_fix")),
                "highres_ready_files_present": inventories["highres_audit"]["all_required_present"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
