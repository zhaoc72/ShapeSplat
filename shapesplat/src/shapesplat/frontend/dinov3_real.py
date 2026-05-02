from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from shapesplat.config import resolve_device
from shapesplat.frontend.dinov3_dependency_check import check_dinov3_dependencies
from shapesplat.frontend.dino_pooling import pool_mask_descriptors


SUPPORTED_DINOV3_MODELS = {
    "dinov3_vits16",
    "dinov3_vitl16",
    "dinov3_vitb16",
    "dinov3_vits16plus",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
}


def _extract_state_dict(payload: Any) -> dict:
    """从常见 checkpoint 包装中取出真正的 state_dict。"""
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "teacher", "backbone"):
            if isinstance(payload.get(key), dict):
                return payload[key]
    return payload


def _clean_state_dict_keys(state: dict) -> dict:
    """清理 DDP / teacher / backbone 前缀，允许本地权重文件名不带官方 hash。"""
    cleaned = {}
    prefixes = ("module.", "teacher.", "backbone.", "model.")
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


class RealDINOv3Wrapper:
    """DINOv3 frozen dense descriptor extractor。

    DINOv3 在 ShapeSplat++ 中不参与训练，只用于提取 dense features，并通过
    mask-guided pooling 得到每个 CO3D visible mask 的 descriptor，供 shape retrieval
    和 identity anchoring 使用。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fcfg = cfg["frontend"]
        self.repo_path = fcfg.get("dino_repo_path")
        self.model_name = fcfg.get("dino_model_name")
        self.checkpoint = fcfg.get("dino_checkpoint")
        self.patch_size = int(fcfg.get("dino_patch_size", 16))
        if self.model_name and self.model_name not in SUPPORTED_DINOV3_MODELS:
            raise ValueError(f"Unsupported DINOv3 model name: {self.model_name}. Supported: {sorted(SUPPORTED_DINOV3_MODELS)}")
        if self.checkpoint is not None and not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"Real DINOv3 checkpoint not found: {Path(self.checkpoint).resolve()}")
        self.device = resolve_device(fcfg.get("dino_device", cfg.get("device", "auto")))
        self.feature_layer = fcfg.get("dino_feature_layer", "last")
        self.input_size = fcfg.get("dino_input_size", None)
        self.l2_normalize = bool(fcfg.get("dino_l2_normalize", True))
        self.model = self._build_model()
        self.model.eval()
        for p in getattr(self.model, "parameters", lambda: [])():
            p.requires_grad_(False)

    def _hub_repo(self) -> str:
        if self.repo_path:
            repo = Path(self.repo_path)
            if not repo.exists():
                raise FileNotFoundError(f"DINOv3 repo path does not exist: {repo.resolve()}")
            return str(repo)
        return "facebookresearch/dinov3"

    def _torch_hub_load(self, **kwargs) -> Any:
        source = "local" if self.repo_path else "github"
        return torch.hub.load(self._hub_repo(), self.model_name, source=source, **kwargs)

    def _build_model(self) -> Any:
        """按多种官方/本地加载形式尝试构建 DINOv3。"""
        if self.model_name is None:
            raise ImportError("real DINOv3 backend needs frontend.dino_model_name.")
        deps = check_dinov3_dependencies()
        if not deps["ok"]:
            raise ImportError(
                "Missing required DINOv3 dependencies. "
                f"missing={deps['missing_required']}. "
                f"Install with: {deps['install_command']}. "
                "Do not run the DINOv3 requirements file if it would overwrite CUDA PyTorch."
            )
        errors: list[str] = []
        checkpoint = str(self.checkpoint) if self.checkpoint else None

        for kwargs in (
            {"weights": checkpoint} if checkpoint else {"pretrained": True},
            {"backbone_weights": checkpoint} if checkpoint else {"pretrained": True},
            {"pretrained": False},
            {},
        ):
            try:
                model = self._torch_hub_load(**kwargs)
                if checkpoint and not ("weights" in kwargs or "backbone_weights" in kwargs):
                    state = _clean_state_dict_keys(_extract_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)))
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if missing:
                        errors.append(f"manual load missing keys sample: {list(missing)[:8]}")
                    if unexpected:
                        errors.append(f"manual load unexpected keys sample: {list(unexpected)[:8]}")
                return model.to(self.device)
            except Exception as exc:
                errors.append(f"torch.hub kwargs={kwargs}: {exc}")

        try:
            transformers = importlib.import_module("transformers")
            AutoModel = getattr(transformers, "AutoModel")
            model = AutoModel.from_pretrained(str(self.checkpoint) if checkpoint else self.model_name)
            return model.to(self.device)
        except Exception as exc:
            errors.append(f"transformers AutoModel: {exc}")

        raise ImportError(
            "Could not build RealDINOv3Wrapper. Check dino_repo_path, dino_model_name, checkpoint compatibility, "
            f"and local DINOv3 repo version. Errors: {errors}"
        )

    def _prepare_input(self, image: torch.Tensor) -> torch.Tensor:
        x = image[None].to(self.device).float().clamp(0, 1)
        if self.input_size is not None:
            x = F.interpolate(x, size=(int(self.input_size), int(self.input_size)), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _get_patch_size(self) -> int:
        """读取 ViT patch size；配置优先，模型属性作为兼容补充。"""
        patch_size = getattr(self, "patch_size", None)
        if patch_size:
            return int(patch_size)
        model_patch = getattr(self.model, "patch_size", None)
        if isinstance(model_patch, (tuple, list)):
            return int(model_patch[0])
        if model_patch is not None:
            return int(model_patch)
        config_patch = getattr(getattr(self.model, "config", None), "patch_size", None)
        return int(config_patch or 16)

    def _forward_dense_model(self, x: torch.Tensor) -> tuple[Any, str]:
        """优先提取 dense patch tokens，最后才尝试普通 forward。

        DINOv3 普通 forward 可能返回 [B,D] 全局图像 embedding；ShapeSplat++ 需要
        每个局部 patch 的 embedding，所以这里优先使用 get_intermediate_layers /
        forward_features 这类 dense API。
        """
        if hasattr(self.model, "get_intermediate_layers"):
            try:
                raw = self.model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)
                return raw, "get_intermediate_layers"
            except Exception as exc:
                self._dense_extraction_errors.append(f"get_intermediate_layers reshape=True: {exc}")
            try:
                raw = self.model.get_intermediate_layers(x, n=1, reshape=False, return_class_token=False)
                return raw, "get_intermediate_layers"
            except Exception as exc:
                self._dense_extraction_errors.append(f"get_intermediate_layers reshape=False: {exc}")

        if hasattr(self.model, "forward_features"):
            try:
                return self.model.forward_features(x), "forward_features"
            except Exception as exc:
                self._dense_extraction_errors.append(f"forward_features: {exc}")

        try:
            return self.model(x, output_hidden_states=True), "forward"
        except TypeError:
            return self.model(x), "forward"

    def _global_embedding_error(self, shape: tuple[int, ...]) -> ValueError:
        """普通 forward 返回全局 embedding 时给出面向实验的明确错误。"""
        return ValueError(
            "DINOv3 returned a global image embedding [B,D]. "
            "ShapeSplat++ requires dense patch features [D,H,W]. "
            "Please use get_intermediate_layers or forward_features. "
            f"Received shape={shape}."
        )

    def _tokens_to_feature_map(
        self,
        tokens: torch.Tensor,
        image_hw: tuple[int, int],
        model_input_hw: tuple[int, int],
    ) -> torch.Tensor:
        """把 ViT patch tokens 转为 dense feature map。

        DINOv3 ViT 输出中可能包含 CLS token 和 register tokens；dense 任务需要局部
        patch embeddings，而不是 global CLS embedding。因此当 token 数量大于 patch
        grid 时，默认取最后 expected 个 token 作为 patch tokens。
        """
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3:
            raise ValueError(f"DINO patch tokens must be [B,L,D], got shape={tuple(tokens.shape)}")
        b, l, d = tokens.shape
        patch_size = self._get_patch_size()
        gh = max(1, int(model_input_hw[0]) // patch_size)
        gw = max(1, int(model_input_hw[1]) // patch_size)
        expected = gh * gw
        if l == expected:
            patch_tokens = tokens
        elif l > expected:
            patch_tokens = tokens[:, -expected:, :]
        else:
            raise ValueError(
                "Cannot reshape DINO patch tokens into dense feature map: "
                f"L={l}, expected={expected}, model_input_hw={model_input_hw}, patch_size={patch_size}."
            )
        fmap = patch_tokens.transpose(1, 2).reshape(b, d, gh, gw)
        return F.interpolate(fmap.float(), size=image_hw, mode="bilinear", align_corners=False)

    def _standardize_dino_output(self, raw_output: Any, image_hw: tuple[int, int], model_input_hw: tuple[int, int]) -> torch.Tensor:
        """标准化 DINOv3 输出为 [D,H,W] dense features。"""
        if torch.is_tensor(raw_output):
            out = raw_output
            if out.ndim == 4:
                fmap = F.interpolate(out.float(), size=image_hw, mode="bilinear", align_corners=False)
            elif out.ndim == 3:
                fmap = self._tokens_to_feature_map(out, image_hw, model_input_hw)
            elif out.ndim == 2:
                if self.cfg.get("frontend", {}).get("dino_allow_global_feature_tiling", False):
                    # 仅用于临时 debug：正式实验不能把全局 embedding 当作局部 dense 特征。
                    self.last_warning = "Using tiled global DINO feature; this is for debugging only."
                    fmap = out[:, :, None, None].expand(out.shape[0], out.shape[1], image_hw[0], image_hw[1]).float()
                else:
                    raise self._global_embedding_error(tuple(out.shape))
            else:
                raise ValueError(f"Unsupported DINO tensor shape: {tuple(out.shape)}")
        elif isinstance(raw_output, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state", "tokens", "features"):
                if key in raw_output:
                    return self._standardize_dino_output(raw_output[key], image_hw, model_input_hw)
            if "hidden_states" in raw_output:
                states = raw_output["hidden_states"]
                state = states[-1] if self.feature_layer == "last" else states[int(self.feature_layer)]
                return self._standardize_dino_output(state, image_hw, model_input_hw)
            raise ValueError(f"Unsupported DINO output dict keys: {list(raw_output.keys())}")
        elif hasattr(raw_output, "last_hidden_state"):
            return self._standardize_dino_output(raw_output.last_hidden_state, image_hw, model_input_hw)
        elif hasattr(raw_output, "hidden_states") and raw_output.hidden_states is not None:
            return self._standardize_dino_output(raw_output.hidden_states[-1], image_hw, model_input_hw)
        elif isinstance(raw_output, (tuple, list)):
            dense_candidates = []
            token_candidates = []
            global_candidates = []
            for item in raw_output:
                if isinstance(item, (tuple, list, dict)) or hasattr(item, "last_hidden_state"):
                    try:
                        return self._standardize_dino_output(item, image_hw, model_input_hw)
                    except ValueError as exc:
                        if "global image embedding" not in str(exc):
                            raise
                if torch.is_tensor(item):
                    if item.ndim == 4:
                        dense_candidates.append(item)
                    elif item.ndim == 3:
                        token_candidates.append(item)
                    elif item.ndim == 2:
                        global_candidates.append(item)
            if dense_candidates:
                return self._standardize_dino_output(dense_candidates[-1], image_hw, model_input_hw)
            if token_candidates:
                return self._standardize_dino_output(token_candidates[-1], image_hw, model_input_hw)
            if global_candidates:
                raise self._global_embedding_error(tuple(global_candidates[-1].shape))
            raise ValueError("Unsupported DINO tuple/list output: no tensor candidates found.")
        else:
            raise TypeError(f"Unsupported DINO output type: {type(raw_output)}")
        fmap = F.interpolate(fmap.float(), size=image_hw, mode="bilinear", align_corners=False)[0]
        if self.l2_normalize:
            fmap = F.normalize(fmap, dim=0)
        return fmap

    @torch.no_grad()
    def extract_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        in_device = image.device
        x = self._prepare_input(image)
        self._dense_extraction_errors: list[str] = []
        raw, mode = self._forward_dense_model(x)
        self.last_extraction_mode = mode
        self.last_model_input_hw = tuple(x.shape[-2:])
        try:
            return self._standardize_dino_output(raw, tuple(image.shape[-2:]), tuple(x.shape[-2:])).to(in_device)
        except ValueError as exc:
            if "global image embedding" in str(exc) and self._dense_extraction_errors:
                raise ValueError(f"{exc} Dense extraction attempts: {self._dense_extraction_errors}") from exc
            raise

    def pool_descriptors(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return pool_mask_descriptors(features, masks, self.cfg)
