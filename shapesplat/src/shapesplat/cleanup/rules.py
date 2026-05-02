from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CLEANUP_RULES = {
    # 中文注释：这些路径永远保护，清理脚本不会把它们作为可执行候选返回。
    "protected_paths": [
        "src",
        "scripts",
        "configs",
        "tests",
        "docs",
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "agents.md",
        ".git",
        "data/co3dv2_single_benchmark",
        "outputs/cache_co3dv2_real_frontend_vits16",
        "outputs/cache_co3dv2_real_frontend_vits16_highres",
        "outputs/check_dinov3_vits16",
        "outputs/check_dinov3_vitl16",
        "outputs/ours_co3dv2_vits16_debug",
        "outputs/ours_co3dv2_vits16_highres",
        "outputs/co3dv2_diagnostics_vits16",
        "outputs/check_gpu_runtime_cuda",
        "outputs/gpu_smoke_cuda",
    ],
    "default_candidate_patterns": {
        "dir_names": [
            "__pycache__",
            ".pytest_cache",
            ".pytest_tmp",
            ".pytest_basetemp",
            "build",
            ".mypy_cache",
            ".ruff_cache",
            "htmlcov",
        ],
        "dir_suffixes": [".egg-info"],
        "file_suffixes": [".pyc"],
        "file_names": [".coverage"],
        "paths": [
            "outputs/minimal",
            "outputs/real_input",
            "outputs/eval_real_input",
            "outputs/check_sam_stub",
            "outputs/check_dino_stub",
            "outputs/check_depth_stub",
            "outputs/check_shape_toy",
            "outputs/check_shape_file",
            "outputs/check_renderer_soft",
            "outputs/artifact_validation_dry",
            "outputs/run_compare_self",
            "outputs/exp_minimal",
            "outputs/exp_stress",
            "outputs/exp_editing",
            "outputs/exp_comparison",
        ],
        "globs": ["outputs/debug*", "outputs/*_debug_old", "dist/*.zip"],
    },
    "default_keep_globs": [
        "data",
        "data/**",
        "examples",
        "examples/**",
        "outputs/cache*",
        "outputs/cache*/**",
        "outputs/ours*",
        "outputs/ours*/**",
        "outputs/co3dv2*",
        "outputs/co3dv2*/**",
        "outputs/final*",
        "outputs/final*/**",
        "outputs/paper*",
        "outputs/paper*/**",
        "outputs/report*",
        "outputs/report*/**",
        "runs",
        "runs/**",
    ],
    "optional_experiment_output_patterns": {
        "globs": ["outputs/cache*", "outputs/ours*", "outputs/co3dv2*", "outputs/final*", "outputs/paper*", "outputs/report*", "runs"],
    },
    "trash_root": ".trash/generated_artifacts",
    "max_depth": 12,
    "notes": "Only generated test/debug artifacts are candidates. Protected research data and source trees are never candidates.",
}


def _merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_cleanup_rules(path: str | Path | None = None) -> dict:
    """加载清理规则；没有配置文件时使用内置安全默认值。"""
    if path is None:
        return deepcopy(DEFAULT_CLEANUP_RULES)
    p = Path(path)
    if not p.exists():
        return deepcopy(DEFAULT_CLEANUP_RULES)
    with open(p, "r", encoding="utf-8") as f:
        user_rules = yaml.safe_load(f) or {}
    return _merge(DEFAULT_CLEANUP_RULES, user_rules)
