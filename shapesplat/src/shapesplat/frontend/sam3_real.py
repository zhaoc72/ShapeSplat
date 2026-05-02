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
    """Optional SAM3 wrapper。

    在 CO3Dv2 流程中，SAM3 只用于 automatic-mask diagnostic；主实验默认使用
    CO3Dv2 file masks，因此 SAM3 加载失败不应影响主实验。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fcfg = cfg["frontend"]
        self.repo_path = fcfg.get("sam3_repo_path")
        checkpoint = fcfg.get("sam3_checkpoint")
        if checkpoint is None:
            raise FileNotFoundError("frontend.sam3_checkpoint is null; real SAM3 backend needs a checkpoint path.")
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Real SAM3 checkpoint not found: {self.checkpoint.resolve()}")
        self.model_type = fcfg.get("sam3_model_type") or "sam3"
        self.prompt_mode = fcfg.get("sam3_prompt_mode", "automatic")
        self.text_prompts = fcfg.get("sam3_text_prompts", ["object"])
        self.score_threshold = float(fcfg.get("sam3_score_threshold", 0.5))
        self.max_masks = int(fcfg.get("sam3_max_masks", 8))
        self.device = resolve_device(fcfg.get("sam3_device", cfg.get("device", "auto")))
        self.predictor = self._build_predictor()

    def _build_predictor(self) -> Any:
        """懒加载 SAM3 代码，避免默认 stub/file-mask 流程依赖 SAM3 package。"""
        errors: list[str] = []
        if self.repo_path:
            import sys

            repo = Path(self.repo_path)
            if not repo.exists():
                raise FileNotFoundError(f"SAM3 repo path does not exist: {repo.resolve()}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
        for module_name in ("sam3", "segment_anything_sam3", "segment_anything"):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
                continue
            try:
                if hasattr(module, "build_sam3"):
                    return module.build_sam3(str(self.checkpoint), model_type=self.model_type, device=str(self.device))
                if hasattr(module, "sam_model_registry") and hasattr(module, "SamAutomaticMaskGenerator"):
                    model_type = self.model_type if self.model_type != "sam3" else "vit_h"
                    model = module.sam_model_registry[model_type](checkpoint=str(self.checkpoint)).to(self.device)
                    return module.SamAutomaticMaskGenerator(model)
            except Exception as exc:
                errors.append(f"{module_name}: checkpoint/model mismatch: {exc}")
                continue
            errors.append(f"{module_name}: no supported builder found")
        raise ImportError(
            "Could not build RealSAM3Wrapper. SAM3 is optional for CO3Dv2 automatic-mask diagnostics; "
            "CO3Dv2 main experiments use file masks. "
            f"checkpoint={self.checkpoint}, model_type={self.model_type}, errors={errors}. "
            "Please check the SAM3 GitHub repo version and checkpoint compatibility."
        )

    def _predict_raw(self, image_np: np.ndarray) -> Any:
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
        raise NotImplementedError("Real SAM3 predictor does not expose generate/predict; adapt RealSAM3Wrapper._predict_raw for this repo version.")

    def _standardize_sam_output(self, raw_output: Any, height: int, width: int, device: torch.device) -> MaskSet:
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
                if not isinstance(item, dict):
                    continue
                m = item.get("segmentation", item.get("mask", item.get("masks", None)))
                if m is None:
                    continue
                mt = normalize_mask(m)
                masks.append(mt)
                scores.append(float(item.get("score", item.get("predicted_iou", item.get("stability_score", 1.0)))))
                boxes.append(mask_to_box(mt))
        if not masks:
            empty = torch.zeros((0, height, width), device=device, dtype=torch.float32)
            return MaskSet(empty, torch.zeros((0,), device=device), torch.zeros((0, 4), device=device))
        masks_t = torch.stack(masks, dim=0).float()
        scores_t = torch.tensor(scores, device=device, dtype=torch.float32)
        boxes_t = torch.stack([b.float() for b in boxes], dim=0)
        return MaskSet(masks_t, scores_t, boxes_t)

    def predict_masks(self, image: torch.Tensor) -> MaskSet:
        _, h, w = image.shape
        image_np = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        raw = self._predict_raw(image_np)
        mask_set = self._standardize_sam_output(raw, h, w, image.device)
        real_cfg = dict(self.cfg)
        real_cfg["frontend"] = dict(self.cfg["frontend"])
        real_cfg["frontend"]["mask_conf_threshold"] = self.score_threshold
        real_cfg["frontend"]["max_num_objects"] = min(int(real_cfg["frontend"].get("max_num_objects", self.max_masks)), self.max_masks)
        return postprocess_masks(mask_set, real_cfg, (h, w))
