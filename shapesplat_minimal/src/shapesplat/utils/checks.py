from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from shapesplat.data.synthetic import make_synthetic_image
from shapesplat.frontend.pipeline import build_frontend
from shapesplat.gaussian.initialization import initialize_scene
from shapesplat.optimization.losses import compute_losses
from shapesplat.renderer.soft_renderer import SoftGaussianRenderer


def _require_file(path: str | Path, kind: str) -> Path:
    """检查输入路径是否存在，给出清晰的中文错误。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{kind} 不存在: {p.resolve()}")
    if not p.is_file():
        raise FileNotFoundError(f"{kind} 不是文件: {p.resolve()}")
    return p


def _is_bad_number(value: Any) -> bool:
    """判断一个 JSON 数值是否为 NaN/Inf。"""
    return isinstance(value, (int, float)) and not math.isfinite(float(value))


def check_loss_log(path: str | Path) -> None:
    """检查 loss_log.json 是否存在、可读取，并确认所有数值都是有限值。

    该检查用于快速发现训练是否发生 NaN/Inf，以及分阶段 loss 是否正常写入日志。
    """
    p = _require_file(path, "loss log")
    with open(p, "r", encoding="utf-8") as f:
        log = json.load(f)
    if not isinstance(log, list):
        raise ValueError(f"loss log 应该是 list，但实际类型是 {type(log).__name__}: {p}")
    if len(log) == 0:
        raise ValueError(f"loss log 为空: {p}")

    print(f"num steps: {len(log)}")
    print(f"first log item: {log[0]}")
    print(f"last log item: {log[-1]}")

    bad_items = []
    stage_totals: dict[str, list[float]] = defaultdict(list)
    for step, item in enumerate(log):
        if not isinstance(item, dict):
            raise ValueError(f"loss log 第 {step} 项不是 dict: {item}")
        stage = item.get("stage")
        total = item.get("total")
        if isinstance(stage, str) and isinstance(total, (int, float)) and math.isfinite(float(total)):
            stage_totals[stage].append(float(total))
        for key, value in item.items():
            if _is_bad_number(value):
                bad_items.append((step, key, value))

    if stage_totals:
        print("stage average loss:")
        for stage, values in stage_totals.items():
            avg = sum(values) / max(1, len(values))
            print(f"  {stage}: {avg:.6f}")

    if bad_items:
        print("NaN / Inf found:")
        for step, key, value in bad_items:
            print(f"  step={step}, item={key}, value={value}")
        raise SystemExit(1)

    print("NaN / Inf: not found")
    print("loss check ok")


def check_checkpoint(path: str | Path) -> None:
    """检查 checkpoint 是否可用 torch.load(map_location='cpu') 重新加载。

    当前最小版本 checkpoint 包含 scene/loss_log/cfg；函数也兼容未来 model/config/optimizer 等键。
    """
    p = _require_file(path, "checkpoint")
    try:
        ckpt = torch.load(p, map_location="cpu")
    except TypeError:
        ckpt = torch.load(p, map_location="cpu")
    except Exception:
        # PyTorch 新版本在 weights_only 默认策略变化时可能拒绝加载 Python 对象；
        # 这是本地可信 checkpoint 检查脚本，因此显式回退到 weights_only=False。
        ckpt = torch.load(p, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint 应该是 dict，但实际类型是 {type(ckpt).__name__}: {p}")

    keys = sorted(ckpt.keys())
    print(f"checkpoint keys: {keys}")
    if "scene" in ckpt:
        print("scene: found")
    if "loss_log" in ckpt:
        print(f"loss_log length: {len(ckpt['loss_log'])}")
    if "cfg" in ckpt:
        print("cfg: found")
    for key in ("model", "config", "optimizer"):
        if key in ckpt:
            print(f"{key}: found")
    print("checkpoint ok")


def check_renderer_shapes(cfg: dict) -> None:
    """从 config 重建 synthetic/front-end/scene/renderer，并检查 renderer 输出 shape。

    该检查不训练，只验证前端输出、Gaussian 初始化和 renderer 张量接口是否仍保持兼容。
    """
    image = make_synthetic_image(int(cfg["image"]["size"]))
    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    renderer = SoftGaussianRenderer(
        front.camera,
        beta_depth=cfg["renderer"]["beta_depth"],
        min_sigma_px=cfg["renderer"]["min_sigma_px"],
        max_sigma_px=cfg["renderer"]["max_sigma_px"],
    )
    out = renderer(scene)

    print(f"rgb shape: {tuple(out.rgb.shape)}")
    print(f"alpha shape: {tuple(out.alpha.shape)}")
    print(f"depth shape: {tuple(out.depth.shape)}")
    print(f"contributions shape: {tuple(out.contributions.shape)}")
    print(f"ownership shape: {tuple(out.ownership.shape)}")
    print(f"bg_ownership shape: {tuple(out.bg_ownership.shape)}")
    print(f"number of masks: {front.masks.shape[0]}")
    print(f"number of objects: {len(scene.objects)}")

    assert out.rgb.ndim == 3
    assert out.rgb.shape[0] == 3
    assert out.alpha.ndim == 2
    assert out.depth.ndim == 2
    assert out.contributions.ndim == 3
    assert out.ownership.ndim == 3
    assert out.contributions.shape[0] == front.masks.shape[0]
    assert out.ownership.shape[0] == front.masks.shape[0]
    assert out.alpha.shape == front.masks.shape[-2:]
    assert out.depth.shape == front.masks.shape[-2:]
    print("renderer shape check ok")


def check_backward(cfg: dict, stage: str = "visible") -> None:
    """检查 loss 是否能反向传播到 Gaussian scene 参数。

    该检查用于确认 renderer 和 loss graph 没有被 detach 或非可微操作意外截断。
    """
    image = make_synthetic_image(int(cfg["image"]["size"]))
    front = build_frontend(image, cfg)
    scene = initialize_scene(front, cfg)
    renderer = SoftGaussianRenderer(
        front.camera,
        beta_depth=cfg["renderer"]["beta_depth"],
        min_sigma_px=cfg["renderer"]["min_sigma_px"],
        max_sigma_px=cfg["renderer"]["max_sigma_px"],
    )
    render = renderer(scene)
    loss, terms = compute_losses(scene, renderer, render, front, cfg, stage=stage)
    loss.backward()

    print(f"total loss: {float(loss.detach().cpu()):.6f}")
    print("loss terms:")
    for key, value in terms.items():
        print(f"  {key}: {value}")

    valid = False
    print("gradients:")
    for name, param in scene.named_parameters():
        if param.grad is None:
            continue
        finite = bool(torch.isfinite(param.grad).all().item())
        grad_sum = float(param.grad.abs().sum().detach().cpu())
        grad_mean = float(param.grad.abs().mean().detach().cpu())
        print(f"  {name}: grad finite={finite}, grad sum={grad_sum:.8f}, grad mean={grad_mean:.8f}")
        valid = valid or (finite and grad_sum > 0)

    if not valid:
        raise RuntimeError("No valid non-zero gradient found. Please check renderer or loss graph.")
    print("backward check ok")
