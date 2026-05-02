from __future__ import annotations

import importlib

import torch

from shapesplat.renderer.camera_conversion import make_real_renderer_camera
from shapesplat.renderer.contract import validate_render_output
from shapesplat.renderer.gaussian_conversion import extract_gaussian_tensors, filter_gaussians_by_object
from shapesplat.renderer.types import RenderOutput


class Real3DGSRendererAdapter:
    """真实 3D Gaussian Splatting renderer 的 compatibility wrapper。

    真实 CUDA renderer 往往只返回 rgb/alpha/depth，而 ShapeSplat++ 的
    scene-coupled ownership optimization 需要 per-object contributions / ownership maps。
    因此如果 renderer 不原生支持 contribution，本 adapter 会预留 object-wise alpha
    fallback：逐物体单独渲染 alpha_n，再归一化为 ownership。

    注意：本文件不在顶层 import diff-gaussian-rasterization / gsplat，避免默认
    stub/soft pipeline 因本地未安装 CUDA renderer 而失败。
    """

    def __init__(self, camera, cfg: dict):
        self.camera = camera
        self.cfg = cfg
        self.rcfg = cfg.get("renderer", {})
        self.gcfg = self.rcfg.get("real_3dgs", {})
        self.available = False
        self.library_name = None
        self.error_message = None
        self._module = None
        self._detect_library()

    def _try_import(self, library: str) -> bool:
        try:
            if library == "diff_gaussian":
                self._module = importlib.import_module("diff_gaussian_rasterization")
            elif library == "gsplat":
                self._module = importlib.import_module("gsplat")
            elif library == "custom":
                module_name = self.rcfg.get("real_renderer_module")
                if not module_name:
                    raise ImportError("renderer.real_renderer_module is not configured for custom renderer")
                self._module = importlib.import_module(module_name)
            else:
                raise ImportError(f"unknown real_3dgs library: {library}")
            self.library_name = library
            self.available = True
            self.error_message = None
            return True
        except Exception as exc:
            self.available = False
            self.error_message = str(exc)
            return False

    def _detect_library(self) -> None:
        requested = str(self.gcfg.get("library", "auto")).lower()
        if requested == "auto":
            errors = []
            for candidate in ("diff_gaussian", "gsplat"):
                if self._try_import(candidate):
                    return
                errors.append(f"{candidate}: {self.error_message}")
            self.error_message = "; ".join(errors) or "no supported real 3DGS library found"
            self.library_name = None
            self.available = False
            return
        self._try_import(requested)

    def scene_to_gaussian_tensors(self, scene) -> dict:
        """导出真实 renderer 需要的 Gaussian tensors。"""

        return extract_gaussian_tensors(scene)

    def make_camera_params(self) -> dict:
        """导出真实 renderer 需要的相机参数。"""

        return make_real_renderer_camera(self.camera, self.cfg)

    def render_all(self, scene) -> RenderOutput:
        """调用真实 renderer 渲染完整 scene。

        这里保留清晰的实现入口。minimal artifact 不捆绑 CUDA renderer，因此当用户
        安装 diff-gaussian / gsplat 后，需要在对应 `_render_with_xxx` 中补齐实际调用。
        """

        if not self.available:
            raise RuntimeError(f"Real 3DGS renderer unavailable: {self.error_message}")
        if self.library_name == "diff_gaussian":
            return self._render_with_diff_gaussian(scene)
        if self.library_name == "gsplat":
            return self._render_with_gsplat(scene)
        if self.library_name == "custom":
            return self._render_with_custom(scene)
        raise RuntimeError(f"Unsupported real 3DGS library: {self.library_name}")

    def _not_implemented(self, library: str):
        raise NotImplementedError(
            f"Real CUDA 3DGS rendering is not implemented for {library} yet. "
            "Use renderer.backend=soft or implement the corresponding _render_with_xxx method."
        )

    def _render_with_diff_gaussian(self, scene) -> RenderOutput:
        self._not_implemented("diff_gaussian")

    def _render_with_gsplat(self, scene) -> RenderOutput:
        self._not_implemented("gsplat")

    def _render_with_custom(self, scene) -> RenderOutput:
        self._not_implemented("custom")

    def render_object(self, scene, object_id: int) -> RenderOutput:
        """单独渲染一个 object。

        真正接入 CUDA renderer 时可以用 filter_gaussians_by_object 后调用低层 rasterizer。
        当前函数保留接口和清晰错误，避免 silent fake rendering。
        """

        _ = filter_gaussians_by_object(self.scene_to_gaussian_tensors(scene), object_id)
        raise NotImplementedError(
            "Object-wise rendering fallback requires a real rasterizer call. "
            "Install/implement a real 3DGS backend or use renderer.backend=soft."
        )

    def compute_object_contributions_by_individual_render(self, scene, all_render: RenderOutput) -> torch.Tensor:
        """逐物体 alpha fallback，构造 contributions。

        如果真实 renderer 不支持 native contribution，后续实现会逐 object render 得到
        alpha_n；contributions=stack(alpha_n)。这能保持 ownership supervision 的输出协议。
        """

        alphas = []
        for obj in scene.objects:
            alphas.append(self.render_object(scene, int(obj.object_id)).alpha)
        if not alphas:
            return torch.zeros((0, self.camera.height, self.camera.width), device=all_render.alpha.device)
        return torch.stack(alphas, dim=0)

    def _as_render_output(self, raw) -> RenderOutput:
        """把真实 renderer 常见的 dict/SimpleNamespace 输出转成 RenderOutput 骨架。"""

        if isinstance(raw, RenderOutput):
            return raw
        if isinstance(raw, dict):
            getter = raw.get
            extras = dict(raw.get("extras", {}))
        else:
            getter = lambda key, default=None: getattr(raw, key, default)
            extras = dict(getattr(raw, "extras", {}))
        rgb = getter("rgb")
        alpha = getter("alpha")
        depth = getter("depth")
        if rgb is None or alpha is None:
            raise RuntimeError("real renderer output must provide at least rgb and alpha")
        if depth is None:
            depth = torch.ones_like(alpha)
        contributions = getter("contributions")
        ownership = getter("ownership")
        bg_ownership = getter("bg_ownership")
        if contributions is None:
            contributions = torch.empty((0, *alpha.shape), device=alpha.device, dtype=alpha.dtype)
        if ownership is None:
            ownership = torch.empty_like(contributions)
        if bg_ownership is None:
            bg_ownership = (1.0 - alpha).clamp(0, 1)
        return RenderOutput(rgb, alpha, depth, contributions, ownership, bg_ownership, extras=extras)

    def _complete_output(self, scene, raw) -> RenderOutput:
        raw = self._as_render_output(raw)
        if raw.contributions.shape[0] != len(scene.objects) or raw.ownership.shape[0] != len(scene.objects):
            contributions = self.compute_object_contributions_by_individual_render(scene, raw)
            bg = (1.0 - raw.alpha).clamp(0, 1)
            denom = (bg + contributions.sum(dim=0)).clamp_min(1.0e-6)
            ownership = contributions / denom[None]
            bg_ownership = bg / denom
            raw = RenderOutput(raw.rgb, raw.alpha, raw.depth, contributions, ownership, bg_ownership, extras=dict(raw.extras))
        raw.extras.update({"renderer_backend": "real_3dgs", "library": self.library_name})
        return raw

    def __call__(self, scene) -> RenderOutput:
        render = self._complete_output(scene, self.render_all(scene))
        contract = validate_render_output(render, len(scene.objects), self.camera.height, self.camera.width, strict=True)
        if not contract["valid"]:
            raise RuntimeError(f"Real 3DGS renderer output violates contract: {contract['errors']}")
        return render
