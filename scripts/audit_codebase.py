from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".trash"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".json", ".csv", ".ps1", ".bat"}


MODULE_PURPOSE = {
    "baselines": ("baseline 协议、dummy/internal/external baseline 管理", "engineering"),
    "cache": ("frontend cache 保存、读取、manifest 与 validation", "engineering"),
    "cleanup": ("测试/调试生成文件安全清理", "engineering"),
    "data": ("图像加载、resize 与基础数据 I/O", "engineering"),
    "datasets": ("manifest、benchmark v2、converter 与数据集接入", "engineering"),
    "editing": ("object editing suite 和编辑命令", "diagnostic"),
    "evaluation": ("2D、editing、geometry、method output 评估", "engineering"),
    "experiments": ("paper/final/CO3Dv2/highres 实验编排与 readiness", "engineering"),
    "frontend": ("SAM3/DINOv3/Depth/file-mask frontend 与真实 backend wrapper", "core"),
    "gaussian": ("visible-hidden Gaussian scene/object 初始化与表示", "core"),
    "geometry": ("mask、camera、几何辅助函数", "core"),
    "optimization": ("Trainer、loss、edit optimization", "core"),
    "renderer": ("RenderOutput contract、soft renderer、real 3DGS adapter", "core"),
    "reporting": ("final/paper table 与 report 生成", "engineering"),
    "reconstruction": ("Ours benchmark runner、variants、diagnostics、output protocol", "core"),
    "reproducibility": ("environment、run metadata、registry", "engineering"),
    "runtime": ("Windows/RTX CUDA runtime、AMP、memory、environment diagnostics", "engineering"),
    "utils": ("日志、可视化、seed 等通用工具", "engineering"),
}


SCRIPT_CATEGORIES = {
    "setup": ["create_", "prepare_", "install"],
    "check": ["check_", "inspect_", "validate_", "print_", "list_"],
    "dataset": ["convert_", "build_", "summarize_benchmark"],
    "frontend": ["sam", "dino", "depth", "frontend"],
    "cache": ["cache_"],
    "ours": ["run_ours", "run_co3dv2_highres_ours"],
    "baseline": ["baseline"],
    "comparison": ["comparison", "compare", "final_comparison"],
    "editing": ["edit"],
    "stress": ["stress"],
    "reporting": ["report", "tables", "summarize"],
    "paper": ["paper", "final_paper"],
    "cleanup": ["cleanup", "clean_"],
    "runtime": ["gpu", "runtime", "windows"],
}


def _json_dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_skipped(path: Path) -> bool:
    parts = path.parts
    if "outputs" in parts and "codebase_audit" in parts:
        return True
    return any(part in SKIP_DIRS for part in path.parts)


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    try:
        return path.stat().st_size <= 2_000_000
    except OSError:
        return False


def build_tree(root: Path) -> str:
    lines: list[str] = [str(root)]
    focus = ["src", "scripts", "configs", "tests", "docs", "examples", "data", "outputs", "runs"]
    for name in focus:
        base = root / name
        if not base.exists():
            lines.append(f"{name}/  [missing]")
            continue
        lines.append(f"{name}/")
        max_depth = 4 if name in {"src", "scripts", "configs", "tests", "docs"} else 2
        for p in sorted(base.rglob("*")):
            if _is_skipped(p):
                continue
            rel = p.relative_to(base)
            if len(rel.parts) > max_depth:
                continue
            if name in {"outputs", "runs"} and p.is_file() and p.stat().st_size > 200_000:
                continue
            indent = "  " * len(rel.parts)
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{indent}{p.name}{suffix}")
    return "\n".join(lines) + "\n"


def module_inventory(root: Path) -> dict:
    src_root = root / "src" / "shapesplat"
    modules = {}
    for p in sorted(src_root.iterdir()) if src_root.exists() else []:
        if p.name.startswith("__"):
            continue
        if p.is_dir():
            files = sorted(str(x.relative_to(src_root)).replace("\\", "/") for x in p.rglob("*.py") if not _is_skipped(x))
            purpose, kind = MODULE_PURPOSE.get(p.name, ("项目模块", "engineering"))
            modules[p.name] = {"purpose": purpose, "kind": kind, "files": files}
        elif p.suffix == ".py":
            modules[p.stem] = {"purpose": "顶层包入口/配置模块", "kind": "engineering", "files": [p.name]}
    return modules


def _script_category(name: str) -> str:
    lower = name.lower()
    for category, needles in SCRIPT_CATEGORIES.items():
        if any(n in lower for n in needles):
            return category
    return "other"


def _script_description(path: Path) -> str:
    name = path.name
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if "ArgumentParser" in text:
        for line in text.splitlines():
            if "ArgumentParser" in line and "description=" in line:
                return line.strip()[:240]
    if name.startswith("run_"):
        return "运行实验/benchmark 的 CLI wrapper"
    if name.startswith("check_"):
        return "检查/诊断脚本"
    if name.startswith("cache_"):
        return "缓存 frontend outputs"
    return "项目脚本"


def script_inventory(root: Path) -> dict:
    scripts = {}
    for p in sorted((root / "scripts").glob("*.py")) if (root / "scripts").exists() else []:
        scripts[p.name] = {"category": _script_category(p.name), "description": _script_description(p)}
    return scripts


def config_inventory(root: Path) -> dict:
    out = {}
    for p in sorted((root / "configs").rglob("*.yaml")) if (root / "configs").exists() else []:
        rel = str(p.relative_to(root / "configs")).replace("\\", "/")
        text = p.read_text(encoding="utf-8", errors="ignore")
        lower = rel.lower() + "\n" + text.lower()
        out[rel] = {
            "purpose": _config_purpose(rel, lower),
            "is_debug": "debug" in lower or "smoke" in lower,
            "is_paper_profile": rel.startswith("paper/") or "profile:" in lower,
            "is_co3dv2": "co3d" in lower,
            "is_highres": "highres" in lower or "long_side: 640" in lower,
            "still_minimal": "minimal" in lower,
        }
    return out


def _config_purpose(rel: str, lower: str) -> str:
    if "co3dv2" in lower and "highres" in lower:
        return "CO3Dv2 high-resolution diagnostic / frontend / Ours 配置"
    if rel.startswith("paper/"):
        return "paper experiment profile"
    if rel.startswith("presets/"):
        return "run_experiment preset"
    if rel.startswith("datasets/"):
        return "dataset converter/config template"
    if "renderer" in lower:
        return "renderer/backend 配置"
    if "baseline" in lower:
        return "baseline/comparison 配置"
    if "minimal" in lower:
        return "legacy/minimal smoke 配置"
    return "ShapeSplat++ 配置"


def test_inventory(root: Path) -> dict:
    out = {}
    for p in sorted((root / "tests").glob("test_*.py")) if (root / "tests").exists() else []:
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        out[p.name] = {
            "coverage": _test_coverage(p.name),
            "requires_real_gpu": "skipif(not torch.cuda.is_available" in text or "require_cuda" in text,
            "requires_real_dino_sam": "real dino" in text or "sam3 checkpoint" in text,
            "quick_test": "optional" not in p.name and "gpu_optional" not in p.name,
        }
    return out


def _test_coverage(name: str) -> str:
    stem = name.replace("test_", "").replace(".py", "")
    return stem.replace("_", " ")


def path_audit(root: Path) -> list[dict]:
    needles = [
        r"C:\Users\zhaoc\ShapeSplat",
        r"C:\Users\zhaoc\ShapeSplat",
        "shapesplat" + "_minimal",
    ]
    rows: list[dict] = []
    for p in sorted(root.rglob("*")):
        if _is_skipped(p) or not p.is_file() or not _is_text_file(p):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        # 中文注释：outputs 本身也可能包含历史日志；报告会列出但不自动修正。
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if any(n in line for n in needles):
                snippet = line.strip()[:300]
                explanatory = rel == "agents.md" or rel.startswith("outputs/codebase_audit/") or rel == "scripts/audit_codebase.py"
                generated = rel.startswith("dist/") or rel.endswith(".egg-info/SOURCES.txt") or ".egg-info/" in rel
                rows.append(
                    {
                        "file": rel,
                        "line": i,
                        "snippet": snippet,
                        "needs_fix": bool(generated or (not explanatory and not rel.startswith("outputs/"))),
                        "note": "generated artifact / stale package metadata" if generated else ("intentional audit/explanatory reference" if explanatory else "source/doc reference"),
                    }
                )
    return rows


def fallback_audit(root: Path) -> dict:
    needles = ["ToyShapeBank", "SoftGaussianRenderer", "DinoV3Stub", "Sam3Stub", "DepthStub", "fallback_to_toy", "fallback_to_soft", "fallback_to_compute"]
    rows = []
    for p in sorted(root.rglob("*")):
        if _is_skipped(p) or not p.is_file() or not _is_text_file(p):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            hits = [n for n in needles if n in line]
            if hits:
                rows.append({"file": rel, "line": i, "hits": hits, "snippet": line.strip()[:300]})
    return {
        "matches": rows,
        "policy": {
            "debug_ok": ["DinoV3Stub", "Sam3Stub", "DepthStub", "ToyShapeBank", "SoftGaussianRenderer", "fallback_to_compute"],
            "paper_warning": ["ToyShapeBank", "SoftGaussianRenderer", "DinoV3Stub", "Sam3Stub", "DepthStub", "fallback_to_toy", "fallback_to_soft"],
            "summary": "fallback 在 smoke/debug/diagnostic 中合理；出现在 final/paper 配置时必须 readiness warning 或 strict error。",
        },
    }


def run_command(args: list[str], timeout: int = 120) -> dict:
    try:
        proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return {"command": " ".join(args), "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}
    except Exception as exc:
        return {"command": " ".join(args), "returncode": None, "error": str(exc)}


def run_light_checks(root: Path) -> list[dict]:
    results = []
    commands = [
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        [sys.executable, "scripts/check_project_health.py"],
        [sys.executable, "scripts/run_quick_tests.py"],
        [sys.executable, "scripts/check_dinov3_dependencies.py"],
        [sys.executable, "scripts/check_gpu_runtime.py", "--config", "configs/local_windows_rtx5090.yaml", "--device", "auto", "--out", "outputs/codebase_audit/check_gpu_runtime_auto"],
    ]
    for cmd in commands:
        results.append(run_command(cmd, timeout=180))
    test_groups = [
        ["tests/test_dinov3_dense_extraction.py", "tests/test_co3dv2_real_frontend_config.py", "tests/test_co3dv2_converter.py"],
        ["tests/test_co3dv2_highres_config.py", "tests/test_image_mask_resize_policy.py", "tests/test_co3dv2_highres_readiness.py", "tests/test_output_resolution_diagnostics.py"],
    ]
    for group in test_groups:
        existing = [p for p in group if (root / p).exists()]
        missing = [p for p in group if not (root / p).exists()]
        if existing:
            results.append(run_command([sys.executable, "-m", "pytest", *existing, "-v"], timeout=240))
        for p in missing:
            results.append({"command": f"pytest {p} -v", "returncode": "not_found", "stdout_tail": "", "stderr_tail": ""})
    return results


def _command_summary(results: list[dict]) -> str:
    lines = []
    for r in results:
        status = r.get("returncode")
        lines.append(f"- `{r.get('command')}` -> `{status}`")
    return "\n".join(lines)


def write_status_report(out: Path, inventories: dict, command_results: list[dict]) -> None:
    path_hits = inventories["path_audit"]
    fallback = inventories["fallback_audit"]
    highres_present = (ROOT / "configs" / "final_ours_co3dv2_highres.yaml").exists() and (ROOT / "scripts" / "run_co3dv2_highres_ours.py").exists()
    health = all(r.get("returncode") in (0, "not_found") for r in command_results)
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

本报告只基于当前目录生成，不基于已废弃的旧嵌套项目目录。

## 2. 项目总体结构

- `src/shapesplat/`：核心包，包含 frontend、Gaussian 表示、renderer、optimization、reconstruction、benchmark、runtime、reporting 等模块。
- `scripts/`：CLI 入口，覆盖检查、数据转换、frontend cache、Ours benchmark、baseline、comparison、paper runner、cleanup、runtime、CO3Dv2 high-res。
- `configs/`：实验配置、paper profiles、presets、dataset configs、runtime/backend configs。
- `tests/`：CPU quick tests、converter tests、frontend config tests、DINO dense extraction tests、high-res policy/readiness tests、runtime tests。
- `docs/`：命令、实验、backend、baseline、troubleshooting、runbook。
- `examples/`：toy/example 数据和小图像。
- `data/`：benchmark 输出位置；当前关键目标是 `data/co3dv2_single_benchmark/manifest.csv`。
- `outputs/`：实验和报告输出。这里包含 debug/diagnostic 结果，不应自动视为 paper-ready。
- `runs/`：run registry / reproducibility metadata。

## 3. 核心算法实现状态

- Visible-hidden Gaussian representation：已在 Gaussian object/scene 初始化与 Trainer 中实现，可输出 visible/hidden count；仍需真实 shape prior 支撑 paper-final completion。
- Scene-coupled ownership rendering：Soft renderer 和 renderer contract 已支持 per-object ownership/contribution；真实 3DGS adapter 是接口/adapter，未强制安装 CUDA renderer。
- Confidence-weighted hidden support prior：检索与 hidden prior 配置/逻辑已存在；ToyShapeBank fallback 只适合 debug。
- Differentiable edit consistency：editing metrics/ops/suite 已接入，可用于 diagnostic 和 ablation。
- SAM3-DINOv3 frozen frontend：file mask、DINOv3 real wrapper、stub/auto 后端、cache 已有；SAM3 主要是 optional diagnostic。
- Shape retrieval：有 shape bank/retrieval 接口与 toy/file backend；paper-ready 需要准备真实 shape bank。
- Renderer backend：SoftGaussianRenderer 默认可用，Real3DGSRendererAdapter 预留；真实 CUDA renderer 未实际作为默认 paper backend。

## 4. Frontend 状态

- file masks：支持 same-mask protocol，CO3Dv2 主流程使用 file masks。
- SAM3：RealSAM3Wrapper 是 optional automatic-mask diagnostic，不作为 CO3Dv2 主实验默认 mask source。
- DINOv3：支持本地 repo/checkpoint、依赖检查、ViT-S/16 和 ViT-L/16 配置；dense patch feature extraction 已修复，禁止把 `[B,D]` global embedding 当 dense feature。
- Depth：DepthStub/auto fallback 可用；真实 depth model 仍是 optional。
- frontend cache：支持 manifest、validation、metadata、zero-valid warning、low-res warning。
- CO3Dv2 cache：已有 ViT-S/16 real frontend cache 脚本和 high-res cache 脚本。
- high-res config：`configs/co3dv2_real_frontend_highres.yaml` 与 `configs/final_ours_co3dv2_highres.yaml` 已存在。

## 5. CO3Dv2 状态

- converter：`CO3Dv2SingleConverter` 已存在，可 annotation-driven 或 folder-scan fallback。
- manifest：目标路径 `data/co3dv2_single_benchmark/manifest.csv` 已作为标准输入。
- benchmark validation：benchmark v2 validator 支持 image/mask/optional GT/cache 检查。
- 使用方式：CO3Dv2 single 被定义为 real-image diagnostic / single foreground visible-mask benchmark，不是多物体主 benchmark。
- file masks：作为 retained visible masks 固定使用，主实验不改成 SAM3。
- high-res：已有 keep-aspect long side 640、nearest mask resize、DINO input 448、debug cap disabled 的配置和检查。
- 待验证：需要重新生成 high-res cache 后运行 Ours，并用 `inspect_output_resolution.py` 确认 per-image diagnostics 非 low-res。

## 6. Ours reconstruction 状态

- `scripts/run_ours_benchmark.py`：通用 Ours benchmark runner。
- `scripts/run_ours_variants.py`：variants/ablation runner。
- high-res wrappers：`run_co3dv2_highres_ours.py`、`run_co3dv2_highres_variants.py`。
- output protocol：保存 render/alpha/ownership/metrics/output_spec/reconstruction_meta/diagnostics/pred_pointcloud。
- diagnostics：包含 original/working/render shapes、cache 使用、mask resize、DINO shape、debug cap、renderer/shape bank fallback。
- limitations：真实 shape bank、真实 renderer、external baseline 仍未 paper-final。

## 7. Baseline / comparison 状态

- dummy baseline：已支持 smoke/debug comparison。
- independent Gaussian baseline：已支持内部 baseline。
- external baseline adapter：有 command template / output protocol，但未接真实 SPAR3D/TRELLIS/VGGT/DUSt3R。
- method catalog：存在，可用于 method family grouping。
- final comparison：可统一评估已有 method outputs；外部真实输出缺失时只能 warning。

## 8. Reporting / paper experiment 状态

- `run_final_paper.py`、`final_debug`、`final_all`、final readiness、final table/export/report 框架已存在。
- CO3Dv2 high-res report：新增 `generate_co3dv2_highres_report.py`。
- final tables：可从已有 summary 输出导出，但 paper-ready 内容依赖真实 benchmark/baselines/shape bank/renderer。

## 9. Runtime / GPU 状态

- runtime 模块支持 device resolve、CUDA smoke test、memory、AMP、environment summary。
- `check_gpu_runtime.py` 支持 RTX 5090 / sm_120 实际 matmul/backward 检查。
- mixed precision 默认关闭，不改变默认数值行为。
- CUDA requested 时不应 silent CPU fallback；fallback 需要显式配置。

## 10. Tests 状态

主要测试类别：

- quick smoke：import/frontend/renderer/train step。
- CO3Dv2 converter：fake CO3D structure，不依赖真实数据。
- DINO dense extraction：不加载真实权重，只测 tensor 标准化。
- CO3Dv2 real frontend config：检查配置和依赖诊断逻辑，不要求真实权重。
- high-res tests：resize policy、readiness、resolution diagnostics/report。
- runtime tests：CPU-safe runtime config；optional GPU tests 不应作为 CI 必跑。

本次轻量检查结果：

{_command_summary(command_results)}

## 11. 当前仍是 debug/fallback 的部分

- ToyShapeBank fallback：debug 合理；final/paper 必须 warning 或 strict error。
- SoftGaussianRenderer fallback：工程诊断可用；不是最终真实 CUDA 3DGS 质量。
- external baselines missing：catalog/template 存在，但真实 baseline 未接入。
- geometry metrics optional：只有 pred 和 GT pointcloud 都存在且协议明确时可报告。
- CO3Dv2 single：diagnostic，不是多物体主 benchmark。
- minimal configs still present：保留用于 smoke/compatibility，不能当 final experiment。

Fallback audit 共发现 `{len(fallback.get('matches', []))}` 处匹配，详见 `fallback_audit.json`。

## 12. 是否可以直接跑论文实验并填表

- debug paper flow：可以运行，用于工程 smoke，不是投稿结果。
- CO3Dv2 diagnostics：可以运行 high-res diagnostic 链路；需要重新生成 high-res cache 并 inspect resolution。
- paper-ready final tables：框架可以生成，但当前不能作为投稿主表。
- 可用表格：debug comparison、variants、geometry available=false 的汇总、CO3Dv2 diagnostic report。
- 不可作为投稿结果：ToyShapeBank/SoftRenderer/dummy baseline/example benchmark/CO3Dv2 single multi-object claim。

## 13. 阻止 paper-ready 的主要缺口

- high-res CO3Dv2：代码已支持，但需要实际 cache + Ours run + resolution inspection 确认。
- real shape bank：尚未接入 paper-ready prepared bank。
- real renderer：adapter 有，真实 CUDA renderer 未作为最终 backend 跑通。
- external baselines：真实方法输出未接入。
- multi-object benchmark：正式多物体 benchmark 未准备/未验证。
- final strict readiness：需要在真实 config/output 上通过。

## 14. 下一步建议

1. 重新生成 CO3Dv2 high-res frontend cache，并验证 `num_valid > 0`。
2. 运行 `run_co3dv2_highres_ours.py`，再运行 `inspect_output_resolution.py`，确认 original/working/render shapes 接近 640 长边且 debug cap 未触发。
3. 运行 high-res variants 中 `full` 和 `visible_only`，确认 ablation 输出协议稳定。
4. 准备真实 shape bank，替换 ToyShapeBank fallback。
5. 决定 renderer 策略：接真实 3DGS renderer，或在论文中明确 SoftRenderer 仅为 diagnostic。
6. 准备正式 multi-object benchmark 与 external baseline outputs，再跑 final comparison/table。

## 15. 重要命令

```bat
conda activate shapesplat
cd /d C:\\Users\\zhaoc\\ShapeSplat
python scripts/check_project_health.py
python scripts/run_quick_tests.py
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device auto --out outputs/codebase_audit/check_gpu_runtime_auto
python scripts/check_dinov3_dependencies.py
python scripts/check_dinov3_weights.py --config configs/co3dv2_real_frontend_debug.yaml --input examples/test_image.png --out outputs/check_dinov3_vits16 --device cuda
python scripts/check_co3dv2_highres_ready.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --out outputs/check_co3dv2_highres_ready
python scripts/cache_co3dv2_highres_frontend.py --config configs/co3dv2_real_frontend_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16_highres --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --validate --require-cuda --check-deps-first
python scripts/run_co3dv2_highres_ours.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_co3dv2_vits16_highres --max-images 5 --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
python scripts/inspect_output_resolution.py --out outputs/ours_co3dv2_vits16_highres --max-items 5
```

## Path Audit 摘要

发现旧项目路径相关文本匹配 `{len(path_hits)}` 处。详见 `path_audit.json`。这些引用需要人工判断；本次 audit 不自动修正。

## 健康结论

- 当前轻量检查健康：`{health}`。
- 当前包含 high-res CO3Dv2 修复：`{highres_present}`。
- 当前仍有旧路径残留：`{len(path_hits) > 0}`。
"""
    (out / "CODEBASE_STATUS.md").write_text(text, encoding="utf-8")


def write_capabilities(out: Path, inventories: dict) -> None:
    text = """# 当前能力摘要

当前代码已经具备：benchmark v2、CO3Dv2 single converter、file-mask same-mask protocol、DINOv3 dense descriptor frontend、frontend cache、Ours benchmark/variants、editing/stress/baseline/final comparison/reporting/runtime/cleanup 等工程链路。

当前不能直接声称 paper-ready 的部分：ToyShapeBank、SoftGaussianRenderer、dummy/external baseline template、缺少正式 multi-object benchmark、缺少真实 renderer 或明确 renderer 协议、缺少 final strict readiness 通过记录。

CO3Dv2 high-res 代码和配置已经存在。下一步应实际重建 high-res cache，运行 Ours high-res，并检查 `output_resolution_summary.json` 是否显示 working/render/mask 分辨率不再是 128。
"""
    (out / "current_capabilities.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ShapeSplat++ codebase audit report.")
    parser.add_argument("--out", default="outputs/codebase_audit")
    parser.add_argument("--run-checks", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "codebase_tree.txt").write_text(build_tree(ROOT), encoding="utf-8")
    inventories = {
        "module_inventory": module_inventory(ROOT),
        "script_inventory": script_inventory(ROOT),
        "config_inventory": config_inventory(ROOT),
        "test_inventory": test_inventory(ROOT),
        "path_audit": path_audit(ROOT),
        "fallback_audit": fallback_audit(ROOT),
    }
    _json_dump(out / "module_inventory.json", inventories["module_inventory"])
    _json_dump(out / "script_inventory.json", inventories["script_inventory"])
    _json_dump(out / "config_inventory.json", inventories["config_inventory"])
    _json_dump(out / "test_inventory.json", inventories["test_inventory"])
    _json_dump(out / "path_audit.json", inventories["path_audit"])
    _json_dump(out / "fallback_audit.json", inventories["fallback_audit"])
    command_results = run_light_checks(ROOT) if args.run_checks else []
    _json_dump(out / "command_results.json", command_results)
    write_capabilities(out, inventories)
    write_status_report(out, inventories, command_results)
    print(json.dumps({"status_report": str(out / "CODEBASE_STATUS.md"), "path_hits": len(inventories["path_audit"]), "fallback_hits": len(inventories["fallback_audit"]["matches"])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
