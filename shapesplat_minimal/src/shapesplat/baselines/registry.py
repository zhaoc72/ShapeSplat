from __future__ import annotations

from shapesplat.baselines.adapters import BaselineAdapter, CommandTemplateAdapter, DummyExternalAdapter


BASELINE_ADAPTERS: dict[str, type[BaselineAdapter]] = {}


def register_adapter(name: str, cls: type[BaselineAdapter]) -> None:
    """注册 baseline adapter。

    后续真实 adapter 如 spar3d、trellis、vggt 可以通过该 registry 接入；当前
    不注册任何未实现的真实 baseline。
    """

    BASELINE_ADAPTERS[name] = cls


def get_adapter(name: str) -> BaselineAdapter:
    """按名称创建 adapter 实例。"""

    if name not in BASELINE_ADAPTERS:
        available = ", ".join(sorted(BASELINE_ADAPTERS))
        raise KeyError(f"Unknown baseline adapter '{name}'. Available adapters: {available}")
    return BASELINE_ADAPTERS[name]()


def list_adapters() -> list[str]:
    """列出已注册 adapter。"""

    return sorted(BASELINE_ADAPTERS.keys())


register_adapter("dummy_external", DummyExternalAdapter)
register_adapter("command_template", CommandTemplateAdapter)

