from __future__ import annotations

import json
from pathlib import Path

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.renderer.backend import build_renderer
from shapesplat.renderer.contract import validate_render_output
from shapesplat.utils.seed import seed_everything
from shapesplat.utils.visualization import save_ownership_argmax


def check_real_renderer(config: str, out: str, backend: str | None = None, input_path: str | None = None, allow_fallback: bool = False, save_visuals: bool = True) -> dict:
    """检查真实 3DGS renderer adapter / soft fallback / RenderOutput contract。"""
    cfg = load_config(config)
    if backend:
        cfg.setdefault("renderer", {})["backend"] = backend
    if allow_fallback:
        cfg.setdefault("renderer", {})["fallback_to_soft"] = True
    seed_everything(int(cfg.get("seed", 0)))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(input_path, size=int(cfg["image"]["size"])) if input_path else make_synthetic_image(int(cfg["image"]["size"]))
    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    renderer = build_renderer(front.camera, cfg)
    render = renderer(scene)
    contract = validate_render_output(render, len(scene.objects), front.camera.height, front.camera.width, strict=True)
    if not contract["valid"]:
        raise AssertionError(contract["errors"])

    stats = {
        "requested_backend": cfg.get("renderer", {}).get("backend", "soft"),
        "renderer_class": renderer.__class__.__name__,
        "available": bool(getattr(renderer, "available", True)),
        "library": getattr(renderer, "library_name", None),
        "fallback_reason": getattr(renderer, "fallback_reason", None),
        "num_objects": len(scene.objects),
        "contract": contract,
    }
    (out_dir / "renderer_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "renderer_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_visuals:
        save_tensor_image(render.rgb, out_dir / "render.png")
        save_tensor_image(render.alpha, out_dir / "alpha.png")
        save_ownership_argmax(render.ownership, out_dir / "ownership_argmax.png")
    print(f"renderer class: {stats['renderer_class']}")
    print(f"requested backend: {stats['requested_backend']}")
    print(f"available: {stats['available']} library: {stats['library']}")
    if stats["fallback_reason"]:
        print(f"fallback: {stats['fallback_reason']}")
    print(f"contract valid: {contract['valid']}")
    return stats
