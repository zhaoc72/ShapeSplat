from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from shapesplat.reporting.io import save_csv_rows
from shapesplat.utils.logging import save_json


@dataclass
class MethodInfo:
    name: str
    family: str
    output_type: str
    native_object_buffers: str | bool
    editable: str | bool
    source: str
    enabled: bool = True
    adapter: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def load_method_catalog(path: str | Path) -> list[MethodInfo]:
    """读取正式实验 method catalog。

    catalog 用于论文表格分组和 baseline 管理；enabled=false 的真实外部方法只是登记，
    不会被默认运行或下载。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"method catalog not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    methods: list[MethodInfo] = []
    for row in data.get("methods", []):
        known_keys = {"name", "family", "output_type", "native_object_buffers", "editable", "source", "enabled", "adapter"}
        kwargs = {k: row.get(k) for k in known_keys if k in row}
        kwargs["metadata"] = {k: v for k, v in row.items() if k not in known_keys}
        methods.append(MethodInfo(**kwargs))
    if not methods:
        raise ValueError(f"method catalog has no methods: {p}")
    return methods


def get_enabled_methods(methods: list[MethodInfo]) -> list[MethodInfo]:
    return [m for m in methods if bool(m.enabled)]


def get_method_by_name(methods: list[MethodInfo], name: str) -> MethodInfo:
    for method in methods:
        if method.name == name:
            return method
    raise KeyError(f"method not found in catalog: {name}")


def group_methods_by_family(methods: list[MethodInfo]) -> dict[str, list[MethodInfo]]:
    groups: dict[str, list[MethodInfo]] = {}
    for method in methods:
        groups.setdefault(method.family, []).append(method)
    return groups


def save_method_catalog_summary(methods: list[MethodInfo], out_path: str | Path) -> None:
    rows = [asdict(m) for m in methods]
    out = Path(out_path)
    if out.suffix.lower() == ".csv":
        save_csv_rows(rows, out)
    else:
        save_json(rows, out)
