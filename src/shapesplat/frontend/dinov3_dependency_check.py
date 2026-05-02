from __future__ import annotations

import importlib.util


REQUIRED_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchmetrics": "torchmetrics",
    "omegaconf": "omegaconf",
    "ftfy": "ftfy",
    "regex": "regex",
    "scikit-learn": "sklearn",
    "submitit": "submitit",
    "termcolor": "termcolor",
}

OPTIONAL_IMPORTS = {
    "transformers": "transformers",
}

INSTALL_COMMAND = "python -m pip install torchmetrics omegaconf ftfy regex scikit-learn submitit termcolor"


def _missing(imports: dict[str, str]) -> list[str]:
    return [package for package, module in imports.items() if importlib.util.find_spec(module) is None]


def check_dinov3_dependencies(require_transformers: bool = False) -> dict:
    """检查 DINOv3 官方 repo 本地加载所需依赖。

    DINOv3 官方代码在 torch.hub 本地加载时常会 import torchmetrics / omegaconf
    等包。这里只做检查，不自动安装；也不要直接 pip install DINOv3 requirements，
    避免覆盖当前已经可用的 CUDA PyTorch。
    """
    missing_required = _missing(REQUIRED_IMPORTS)
    missing_optional = _missing(OPTIONAL_IMPORTS)
    if require_transformers and "transformers" in missing_optional:
        missing_required.append("transformers")
        missing_optional = [x for x in missing_optional if x != "transformers"]
    warnings: list[str] = []
    if "torchmetrics" in missing_required:
        warnings.append("torchmetrics is a common DINOv3 repo dependency; install missing deps before loading local DINOv3.")
    if missing_required:
        warnings.append("Install only the missing DINOv3 helper dependencies; do not reinstall torch from a repo requirements file.")
    return {
        "ok": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "install_command": INSTALL_COMMAND,
        "warnings": warnings,
    }
