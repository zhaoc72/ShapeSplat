from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from shapesplat.utils.config_override import apply_overrides


def load_ours_variants(path: str | Path) -> list[dict]:
    """读取 Ours 内部方法变体。

    variant 是论文消融中的主方法开关，不是外部 baseline；这里保留 dotted-key
    overrides，方便和已有 ablation 配置系统共用。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ours variants config not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    variants = data.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"Ours variants config has no variants: {p}")
    return variants


def get_variant_by_name(variants: list[dict], name: str) -> dict:
    """按名字取一个 Ours 变体。"""
    for variant in variants:
        if variant.get("name") == name:
            return variant
    raise KeyError(f"Ours variant not found: {name}")


def apply_variant_overrides(cfg: dict, variant: dict) -> dict:
    """把 variant overrides 应用到 cfg，返回深拷贝后的新配置。"""
    out = apply_overrides(deepcopy(cfg), variant.get("overrides", {}))
    name = variant.get("name", out.get("ours", {}).get("variant", "full"))
    out.setdefault("ours", {})
    out["ours"]["variant"] = name
    out["ablation_name"] = name
    return out
