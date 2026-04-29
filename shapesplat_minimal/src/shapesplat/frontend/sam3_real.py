from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from shapesplat.config import resolve_device
from shapesplat.frontend.mask_postprocess import postprocess_masks
from shapesplat.frontend.types import MaskSet
from shapesplat.geometry.masks import mask_to_box


class RealSAM3Wrapper:
    """可选真实 SAM3 wrapper。

    当前项目不依赖真实 SAM3 checkpoint；没有安装 SAM3 或没有 checkpoint 时，
    默认 stub/auto 模式仍可正常工作。真实 SAM3 API 可能因本地安装方式不同需要
    对 _build_predictor / _predict_raw 做少量适配。输出始终是 visible instance masks，
    不是 amodal masks。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fcfg = cfg["frontend"]
        checkpoint = fcfg.get("sam3_checkpoint")
        if checkpoint is None:
            raise FileNotFoundError("frontend.sam3_checkpoint is null; real SAM3 backend needs a checkpoint path.")
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"真实 SAM3 checkpoint 不存在: {self.checkpoint.resolve()}")
        self.model_type = fcfg.get("sam3_model_type")
        self.prompt_mode = fcfg.get("sam3_prompt_mode", "automatic")
        self.text_prompts = fcfg.get("sam3_text_prompts", ["object"])
        self.score_threshold = float(fcfg.get("sam3_score_threshold", 0.5))
        self.max_masks = int(fcfg.get("sam3_max_masks", 8))
        self.device = resolve_device(fcfg.get("sam3_device", cfg.get("device", "auto")))
        self.predictor = self._build_predictor()

    def _build_predictor(self) -> Any:
        """延迟导入真实 SAM3 依赖，避免默认 stub 模式 import 失败。

        这里提供几个常见入口的适配占位。若你的 SAM3 安装 API 不同，只需在本函数里
        构造一个带 predict/generate/set_image 等方法的 predictor。
        """
        errors = []
        for module_name in ("sam3", "segment_anything", "segment_anything_sam3"):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
                continue
            if hasattr(module, "build_sam3"):
                model = module.build_sam3(str(self.checkpoint), model_type=self.model_type, device=str(self.device))
                return model
            if hasattr(module, "sam_model_registry") and hasattr(module, "SamAutomaticMaskGenerator"):
                model_type = self.model_type or "vit_h"
                model = module.sam_model_registry[model_type](checkpoint=str(self.checkpoint)).to(self.device)
                return module.SamAutomaticMaskGenerator(model)
            errors.append(f"{module_name}: no supported builder found")
        raise ImportError(
            "无法构建真实 SAM3 predictor。请安装 SAM3 并按本地 API 适配 RealSAM3Wrapper._build_predictor。"
            f" 尝试记录: {errors}"
        )

    def _predict_raw(self, image_np: np.ndarray) -> Any:
        """调用真实 predictor。

        支持 automatic proposals 或 text prompts 两种模式；具体真实 API 若不同，可在这里适配。
        """
        predictor = self.predictor
        if hasattr(predictor, "generate"):
            return predictor.generate(image_np)
        if hasattr(predictor, "predict"):
            if self.prompt_mode == "text":
                return predictor.predict(image_np, text_prompts=self.text_prompts)
            return predictor.predict(image_np)
        if hasattr(predictor, "set_image") and hasattr(predictor, "predict"):
            predictor.set_image(image_np)
            return predictor.predict()
        raise NotImplementedError("真实 SAM3 predictor 没有 generate/predict 接口，请适配 _predict_raw。")

    def _standardize_sam_output(self, raw_output: Any, height: int, width: int, device: torch.device) -> MaskSet:
        """将真实 SAM3 的不同输出格式统一成 MaskSet。

        支持两类常见返回：
        1. list[dict]，每项包含 segmentation/mask、score/predicted_iou、bbox/box；
        2. dict，包含 masks / scores / boxes。
        """
        def normalize_mask(mask_like: Any) -> torch.Tensor:
            mt = torch.as_tensor(mask_like, device=device).float()
            while mt.ndim > 2:
                mt = mt.squeeze(0)
            if mt.shape != (height, width):
                mt = F.interpolate(mt[None, None], size=(height, width), mode="nearest")[0, 0]
            return (mt > 0.5).float()

        masks, scores, boxes = [], [], []
        if isinstance(raw_output, dict):
            raw_masks = raw_output.get("masks", raw_output.get("segmentations", []))
            raw_scores = raw_output.get("scores", raw_output.get("confidences", None))
            raw_boxes = raw_output.get("boxes", raw_output.get("bboxes", None))
            for i, m in enumerate(raw_masks):
                mt = normalize_mask(m)
                masks.append(mt)
                scores.append(float(raw_scores[i]) if raw_scores is not None else 1.0)
                boxes.append(torch.as_tensor(raw_boxes[i], device=device).float() if raw_boxes is not None else mask_to_box(mt))
        elif isinstance(raw_output, list):
            for item in raw_output:
                if isinstance(item, dict):
                    m = item.get("segmentation", item.get("mask", item.get("masks", None)))
                    if m is None:
                        continue
                    mt = normalize_mask(m)
                    masks.append(mt)
                    scores.append(float(item.get("score", item.get("predicted_iou", item.get("stability_score", 1.0)))))
                    box = item.get("bbox", item.get("box", None))
                    if box is not None and len(box) == 4:
                        bt = torch.as_tensor(box, device=device).float()
                        # SAM/SAM2 常见 bbox 是 xywh；若看起来像 xywh，则转 xyxy。
                        if bt[2] <= width and bt[3] <= height and bt[2] > bt[0] and bt[3] > bt[1]:
                            boxes.append(bt)
                        else:
                            boxes.append(torch.tensor([bt[0], bt[1], bt[0] + bt[2], bt[1] + bt[3]], device=device))
                    else:
                        boxes.append(mask_to_box(mt))
        if not masks:
            empty = torch.zeros((0, height, width), device=device, dtype=torch.float32)
            return MaskSet(empty, torch.zeros((0,), device=device), torch.zeros((0, 4), device=device))
        masks_t = torch.stack([(m > 0.5).float().reshape(height, width) for m in masks], dim=0).float()
        scores_t = torch.tensor(scores, device=device, dtype=torch.float32)
        boxes_t = torch.stack([b.float() for b in boxes], dim=0)
        return MaskSet(masks_t, scores_t, boxes_t)

    def predict_masks(self, image: torch.Tensor) -> MaskSet:
        """对 [3,H,W]、[0,1] RGB tensor 预测 visible instance masks。"""
        _, h, w = image.shape
        image_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        raw = self._predict_raw(image_np)
        mask_set = self._standardize_sam_output(raw, h, w, image.device)
        # real SAM3 先用 sam3_score_threshold，再复用全局 mask_conf_threshold/min_area/max_objects。
        real_cfg = dict(self.cfg)
        real_cfg["frontend"] = dict(self.cfg["frontend"])
        real_cfg["frontend"]["mask_conf_threshold"] = self.score_threshold
        real_cfg["frontend"]["max_num_objects"] = min(int(real_cfg["frontend"].get("max_num_objects", self.max_masks)), self.max_masks)
        return postprocess_masks(mask_set, real_cfg, (h, w))
