from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from shapesplat.config import resolve_device


class RealDepthWrapper:
    """可选真实 monocular depth wrapper。

    当前项目不强依赖真实 depth checkpoint。真实 API 可能需要根据本地安装方式
    在 _build_model / _forward_model 中稍作适配。depth 只作为 weak initialization /
    layout cue，不是 oracle geometry。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fcfg = cfg["frontend"]
        self.model_name = fcfg.get("depth_model_name")
        self.checkpoint = fcfg.get("depth_checkpoint")
        if self.checkpoint is not None and not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"真实 depth checkpoint 不存在: {Path(self.checkpoint).resolve()}")
        self.device = resolve_device(fcfg.get("depth_device", cfg.get("device", "auto")))
        self.input_size = fcfg.get("depth_input_size", None)
        self.inverse = bool(fcfg.get("depth_inverse", False))
        self.min_valid = float(fcfg.get("depth_min_valid", 1e-6))
        self.model = self._build_model()
        self.model.eval()
        for p in getattr(self.model, "parameters", lambda: [])():
            p.requires_grad_(False)

    def _build_model(self) -> Any:
        """延迟导入真实 depth 依赖，支持本地 wrapper / torch hub / transformers 常见路径。"""
        errors = []
        if self.model_name is None and self.checkpoint is None:
            raise ImportError("real depth backend 需要 frontend.depth_model_name 或 frontend.depth_checkpoint；也可以使用 stub/auto。")

        # A. 用户本地 wrapper：模块需要提供 build_depth_model。
        if self.model_name:
            try:
                module = importlib.import_module(self.model_name)
                if hasattr(module, "build_depth_model"):
                    return module.build_depth_model(checkpoint=self.checkpoint, device=str(self.device))
            except Exception as exc:
                errors.append(f"local wrapper {self.model_name}: {exc}")

        # B. torch hub / transformers 风格。
        if self.model_name:
            try:
                model = torch.hub.load("isl-org/ZoeDepth", self.model_name, pretrained=self.checkpoint is None)
                if self.checkpoint is not None:
                    state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
                    model.load_state_dict(state.get("model", state), strict=False)
                return model.to(self.device)
            except Exception as exc:
                errors.append(f"torch.hub depth: {exc}")

        try:
            transformers = importlib.import_module("transformers")
            AutoModel = getattr(transformers, "AutoModel")
            model = AutoModel.from_pretrained(str(self.checkpoint) if self.checkpoint else self.model_name)
            return model.to(self.device)
        except Exception as exc:
            errors.append(f"transformers AutoModel: {exc}")

        raise ImportError("无法构建真实 depth 模型。请安装模型依赖，或使用 depth_backend=stub/auto。尝试记录: " + str(errors))

    def _prepare_input(self, image: torch.Tensor) -> torch.Tensor:
        x = image[None].to(self.device).float().clamp(0, 1)
        if self.input_size is not None:
            x = F.interpolate(x, size=(int(self.input_size), int(self.input_size)), mode="bilinear", align_corners=False)
        return x

    def _forward_model(self, x: torch.Tensor) -> Any:
        if hasattr(self.model, "infer"):
            return self.model.infer(x)
        if hasattr(self.model, "infer_pil"):
            return self.model(x)
        return self.model(x)

    def _standardize_depth_output(self, raw_output: Any, image_hw: tuple[int, int]) -> torch.Tensor:
        """把不同 depth model 输出统一成 [H,W] 并 resize 到原图大小。"""
        out = raw_output
        if isinstance(out, dict):
            if "depth" in out:
                out = out["depth"]
            elif "predicted_depth" in out:
                out = out["predicted_depth"]
            elif "disparity" in out:
                out = out["disparity"]
            else:
                raise ValueError(f"无法识别 depth 输出 dict keys: {list(out.keys())}")
        elif hasattr(out, "predicted_depth"):
            out = out.predicted_depth
        elif isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            out = torch.as_tensor(out)
        out = out.float().to(self.device)
        while out.ndim > 2:
            out = out.squeeze(0)
        if out.ndim != 2:
            raise ValueError(f"不支持的 depth 输出形状: {tuple(out.shape)}")
        out = F.interpolate(out[None, None], size=image_hw, mode="bilinear", align_corners=False)[0, 0]
        if self.inverse:
            # 不同模型深度方向可能不同，用户需要确认；这里按 disparity -> depth-like 处理。
            out = 1.0 / out.clamp_min(self.min_valid)
        return out

    @torch.no_grad()
    def predict_depth(self, image: torch.Tensor) -> torch.Tensor:
        """输入 [3,H,W] RGB tensor，输出 [H,W] raw depth，返回到输入 image device。"""
        in_device = image.device
        raw = self._forward_model(self._prepare_input(image))
        return self._standardize_depth_output(raw, tuple(image.shape[-2:])).to(in_device)
