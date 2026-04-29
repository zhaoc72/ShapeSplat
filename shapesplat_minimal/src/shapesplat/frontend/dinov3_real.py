from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from shapesplat.config import resolve_device
from shapesplat.frontend.dino_pooling import pool_mask_descriptors


class RealDINOv3Wrapper:
    """可选真实 DINOv3 wrapper。

    当前项目不强依赖真实 DINOv3 checkpoint。真实 API 可能需要根据本地安装方式
    在 _build_model / _forward_model 中稍作适配。DINOv3 在本项目中始终是 frozen
    dense descriptor extractor，不参与训练、不反传梯度。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fcfg = cfg["frontend"]
        self.model_name = fcfg.get("dino_model_name")
        self.checkpoint = fcfg.get("dino_checkpoint")
        if self.checkpoint is not None and not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"真实 DINOv3 checkpoint 不存在: {Path(self.checkpoint).resolve()}")
        self.device = resolve_device(fcfg.get("dino_device", cfg.get("device", "auto")))
        self.feature_layer = fcfg.get("dino_feature_layer", "last")
        self.input_size = fcfg.get("dino_input_size", None)
        self.l2_normalize = bool(fcfg.get("dino_l2_normalize", True))
        self.model = self._build_model()
        self.model.eval()
        for p in getattr(self.model, "parameters", lambda: [])():
            p.requires_grad_(False)

    def _build_model(self) -> Any:
        """延迟导入真实 DINOv3 依赖，支持 torch hub 与 HuggingFace 两类常见路径。"""
        errors = []
        if self.model_name is None and self.checkpoint is None:
            raise ImportError("real DINOv3 backend 需要 frontend.dino_model_name 或 frontend.dino_checkpoint；也可以使用 stub/auto。")

        # A. torch hub / facebookresearch 风格。真实 repo 名称可能需按本地安装调整。
        if self.model_name is not None:
            try:
                model = torch.hub.load("facebookresearch/dinov3", self.model_name, pretrained=self.checkpoint is None)
                if self.checkpoint is not None:
                    state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
                    model.load_state_dict(state.get("model", state), strict=False)
                return model.to(self.device)
            except Exception as exc:
                errors.append(f"torch.hub facebookresearch/dinov3: {exc}")

        # B. transformers / HuggingFace 风格。
        try:
            transformers = importlib.import_module("transformers")
            AutoModel = getattr(transformers, "AutoModel")
            if self.checkpoint is not None:
                model = AutoModel.from_pretrained(str(self.checkpoint))
            else:
                model = AutoModel.from_pretrained(self.model_name)
            return model.to(self.device)
        except Exception as exc:
            errors.append(f"transformers AutoModel: {exc}")

        raise ImportError(
            "无法构建真实 DINOv3 模型。请安装 DINOv3/transformers，或使用 dino_backend=stub/auto。"
            f" 尝试记录: {errors}"
        )

    def _prepare_input(self, image: torch.Tensor) -> torch.Tensor:
        x = image[None].to(self.device).float().clamp(0, 1)
        if self.input_size is not None:
            x = F.interpolate(x, size=(int(self.input_size), int(self.input_size)), mode="bilinear", align_corners=False)
        # 常见 DINO/ImageNet normalization；如果本地 wrapper 自带 preprocess，可在这里替换。
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _forward_model(self, x: torch.Tensor) -> Any:
        """真实模型前向。不同 API 可在这里适配 feature layer。"""
        try:
            return self.model(x, output_hidden_states=True)
        except TypeError:
            return self.model(x)

    def _standardize_dino_output(self, raw_output: Any, image_hw: tuple[int, int]) -> torch.Tensor:
        """把真实 DINOv3 的输出统一成 [D,H,W] dense feature map。"""
        h, w = image_hw
        out = raw_output
        if isinstance(out, dict):
            if "last_hidden_state" in out:
                out = out["last_hidden_state"]
            elif "hidden_states" in out:
                states = out["hidden_states"]
                out = states[-1] if self.feature_layer == "last" else states[int(self.feature_layer)]
            elif "features" in out:
                out = out["features"]
            else:
                raise ValueError(f"无法识别 DINO 输出 dict keys: {list(out.keys())}")
        elif hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            out = out.hidden_states[-1]
        elif isinstance(out, (tuple, list)):
            out = out[-1]

        if not torch.is_tensor(out):
            raise TypeError(f"无法把 DINO 输出转成 tensor: {type(out)}")

        if out.ndim == 4:
            fmap = out
        elif out.ndim == 3:
            # [B,L,D] patch tokens。若有 CLS token，尝试去掉第一个 token。
            b, l, d = out.shape
            grid_l = l
            tokens = out
            side = int(math.sqrt(grid_l))
            if side * side != grid_l and l > 1:
                grid_l = l - 1
                side = int(math.sqrt(grid_l))
                tokens = out[:, 1:, :]
            if side * side != grid_l:
                raise ValueError(f"DINO patch token 数量不是平方数，无法 reshape: L={l}")
            fmap = tokens.transpose(1, 2).reshape(b, d, side, side)
        else:
            raise ValueError(f"不支持的 DINO 输出维度: {tuple(out.shape)}")

        fmap = F.interpolate(fmap.float(), size=(h, w), mode="bilinear", align_corners=False)[0]
        if self.l2_normalize:
            fmap = F.normalize(fmap, dim=0)
        return fmap

    @torch.no_grad()
    def extract_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        """输入 [3,H,W] RGB tensor，输出 [D,H,W] dense features，返回到输入 image device。"""
        in_device = image.device
        x = self._prepare_input(image)
        raw = self._forward_model(x)
        return self._standardize_dino_output(raw, tuple(image.shape[-2:])).to(in_device)

    def pool_descriptors(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """使用统一 mask-guided pooling 得到 [N,D] descriptor。"""
        with torch.no_grad():
            return pool_mask_descriptors(features, masks, self.cfg)
