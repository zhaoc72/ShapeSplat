from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def validate_baseline_output_dir(
    output_dir: str | Path,
    expected_num_objects: int | None = None,
    image_hw: tuple[int, int] | None = None,
    strict: bool = False,
) -> dict:
    """验证外部 baseline 输出是否满足统一 output protocol。

    外部方法只要能输出 render/alpha/ownership 或 object_i_alpha，就可以被后续
    loader 和统一 metrics 评估；本函数只做格式检查，不评价算法质量。
    """

    root = Path(output_dir)
    warnings: list[str] = []
    errors: list[str] = []
    found: dict[str, str | list[str] | None] = {}

    if not root.exists():
        return {"valid": False, "warnings": [], "errors": [f"output dir not found: {root}"], "found_files": {}}

    render = _first_existing(root, ["render.png", "render_final.png"])
    alpha = _first_existing(root, ["alpha.png", "alpha_final.png"])
    ownership = root / "ownership.npy"
    object_alphas = sorted(root.glob("object_*_alpha.png"))
    metrics = root / "metrics.json"
    spec = root / "output_spec.json"

    found.update(
        {
            "render": str(render) if render else None,
            "alpha": str(alpha) if alpha else None,
            "ownership": str(ownership) if ownership.exists() else None,
            "object_alphas": [str(p) for p in object_alphas],
            "metrics": str(metrics) if metrics.exists() else None,
            "output_spec": str(spec) if spec.exists() else None,
        }
    )

    if render is None:
        (errors if strict else warnings).append("missing render.png/render_final.png")
    else:
        try:
            Image.open(render).verify()
        except Exception as exc:
            errors.append(f"render image cannot be opened: {exc}")

    if alpha is None:
        (errors if strict else warnings).append("missing alpha.png/alpha_final.png")
    else:
        try:
            arr = np.asarray(Image.open(alpha).convert("L"), dtype=np.float32) / 255.0
            if image_hw is not None and tuple(arr.shape) != tuple(image_hw):
                errors.append(f"alpha shape {tuple(arr.shape)} != expected {tuple(image_hw)}")
            if not np.isfinite(arr).all():
                errors.append("alpha has NaN/Inf")
            if arr.min() < -1e-4 or arr.max() > 1.0 + 1e-4:
                warnings.append("alpha range outside [0,1]")
        except Exception as exc:
            errors.append(f"alpha image cannot be opened: {exc}")

    if ownership.exists():
        try:
            arr = np.load(ownership)
            if arr.ndim != 3:
                errors.append(f"ownership.npy must be [N,H,W], got {arr.shape}")
            else:
                if expected_num_objects is not None and arr.shape[0] != int(expected_num_objects):
                    errors.append(f"ownership N={arr.shape[0]} != expected {expected_num_objects}")
                if image_hw is not None and tuple(arr.shape[-2:]) != tuple(image_hw):
                    errors.append(f"ownership HW={tuple(arr.shape[-2:])} != expected {tuple(image_hw)}")
            if not np.isfinite(arr).all():
                errors.append("ownership.npy has NaN/Inf")
        except Exception as exc:
            errors.append(f"ownership.npy cannot be loaded: {exc}")
    elif object_alphas:
        if expected_num_objects is not None and len(object_alphas) != int(expected_num_objects):
            errors.append(f"object_i_alpha count={len(object_alphas)} != expected {expected_num_objects}")
        for p in object_alphas:
            try:
                Image.open(p).verify()
            except Exception as exc:
                errors.append(f"{p.name} cannot be opened: {exc}")
    else:
        (errors if strict else warnings).append("missing ownership.npy or object_i_alpha.png")

    return {"valid": len(errors) == 0, "warnings": warnings, "errors": errors, "found_files": found}

