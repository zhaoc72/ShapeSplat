from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.data.image_io import load_image, save_tensor_image
from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.renderer.backend import build_renderer
from shapesplat.renderer.contract import validate_render_output
from shapesplat.utils.seed import seed_everything
from shapesplat.utils.visualization import save_ownership_argmax


def _check_render_output(out, num_objects: int, height: int, width: int) -> dict:
    """检查 renderer 输出是否满足 ShapeSplat++ 的标准 RenderOutput contract。"""
    contract = validate_render_output(out, num_objects, height, width, strict=True)
    if not contract["valid"]:
        raise AssertionError(contract["errors"])
    assert out.rgb.shape == (3, height, width)
    assert out.alpha.shape == (height, width)
    assert out.depth.shape == (height, width)
    assert out.contributions.shape == (num_objects, height, width)
    assert out.ownership.shape == (num_objects, height, width)
    assert out.bg_ownership.shape == (height, width)
    ownership_sum = out.bg_ownership + out.ownership.sum(dim=0)
    assert torch.isfinite(ownership_sum).all()
    assert float((ownership_sum - 1.0).abs().mean().detach().cpu()) < 1e-3
    return {
        "rgb_shape": list(out.rgb.shape),
        "alpha_shape": list(out.alpha.shape),
        "depth_shape": list(out.depth.shape),
        "contributions_shape": list(out.contributions.shape),
        "ownership_shape": list(out.ownership.shape),
        "ownership_sum_min": float(ownership_sum.min().detach().cpu()),
        "ownership_sum_max": float(ownership_sum.max().detach().cpu()),
        "ownership_sum_mean": float(ownership_sum.mean().detach().cpu()),
        "alpha_min": float(out.alpha.min().detach().cpu()),
        "alpha_max": float(out.alpha.max().detach().cpu()),
        "alpha_mean": float(out.alpha.mean().detach().cpu()),
    }


def check_renderer_backend(config: str, backend: str | None, out: str, input_path: str | None) -> None:
    """构建 renderer backend 并检查 RenderOutput shape 与 ownership 归一性。

    这个脚本只检查 renderer contract，不评估真实重建质量。真实 CUDA renderer
    后续接入时也应通过这条检查，尤其要返回 per-object contribution maps。
    """
    cfg = load_config(config)
    if backend is not None:
        cfg.setdefault("renderer", {})["backend"] = backend
    seed_everything(int(cfg["seed"]))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_input = input_path or cfg["image"].get("input_path")
    if resolved_input:
        image = load_image(resolved_input, size=int(cfg["image"]["size"]))
    else:
        image = make_synthetic_image(int(cfg["image"]["size"]))
    save_tensor_image(image, out_dir / "input.png")

    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    renderer = build_renderer(front.camera, cfg)
    render = renderer(scene)

    stats = _check_render_output(render, len(scene.objects), front.camera.height, front.camera.width)
    stats.update(
        {
            "backend": cfg["renderer"].get("backend", "soft"),
            "renderer_class": renderer.__class__.__name__,
            "num_objects": len(scene.objects),
        }
    )
    save_tensor_image(render.rgb, out_dir / "render.png")
    save_tensor_image(render.alpha, out_dir / "alpha.png")
    save_ownership_argmax(render.ownership, out_dir / "ownership_argmax.png")
    (out_dir / "renderer_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"backend: {stats['backend']}")
    print(f"renderer class: {stats['renderer_class']}")
    print(f"rgb shape: {tuple(render.rgb.shape)}")
    print(f"alpha shape: {tuple(render.alpha.shape)}")
    print(f"depth shape: {tuple(render.depth.shape)}")
    print(f"contributions shape: {tuple(render.contributions.shape)}")
    print(f"ownership shape: {tuple(render.ownership.shape)}")
    print(
        "ownership sum min/max/mean: "
        f"{stats['ownership_sum_min']:.6f}/"
        f"{stats['ownership_sum_max']:.6f}/"
        f"{stats['ownership_sum_mean']:.6f}"
    )
    print(f"alpha min/max/mean: {stats['alpha_min']:.6f}/{stats['alpha_max']:.6f}/{stats['alpha_mean']:.6f}")
    print("renderer backend check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ShapeSplat++ renderer backend contract.")
    parser.add_argument("--config", default="configs/minimal.yaml")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--out", default="outputs/check_renderer_backend")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    check_renderer_backend(args.config, args.backend, args.out, args.input)


if __name__ == "__main__":
    main()
