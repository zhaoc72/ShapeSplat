from __future__ import annotations

import importlib
from pathlib import Path


def _base(name: str, requested: str) -> dict:
    return {
        "backend": name,
        "requested": requested,
        "available": False,
        "will_fallback": False,
        "fallback_target": None,
        "errors": [],
        "warnings": [],
        "details": {},
    }


def _checkpoint_ok(value) -> tuple[bool, str | None]:
    if value in (None, "", "null"):
        return False, "checkpoint/model path is not configured"
    path = Path(value)
    if not path.exists():
        return False, f"path does not exist: {path}"
    return True, None


def detect_sam_capability(cfg: dict) -> dict:
    """轻量检测 SAM backend 是否具备本地真实接入条件。

    这里只检查配置、wrapper import 和 checkpoint 路径，不执行 SAM 推理。
    """

    fcfg = cfg.get("frontend", {})
    requested = str(fcfg.get("sam_backend", "stub")).lower()
    out = _base("sam", requested)
    if requested == "stub":
        out["available"] = True
        return out
    try:
        importlib.import_module("shapesplat.frontend.sam3_real")
    except Exception as exc:
        out["errors"].append(f"RealSAM3Wrapper import failed: {exc}")
    ok, err = _checkpoint_ok(fcfg.get("sam3_checkpoint"))
    if not ok:
        out["errors"].append(err)
    out["available"] = not out["errors"]
    if requested == "auto" and not out["available"]:
        out["will_fallback"] = True
        out["fallback_target"] = "stub"
        out["warnings"].append("SAM auto backend will fallback to stub")
    return out


def detect_dino_capability(cfg: dict) -> dict:
    """检测 DINO backend；不下载 torch hub 或 HuggingFace 权重。"""

    fcfg = cfg.get("frontend", {})
    requested = str(fcfg.get("dino_backend", "stub")).lower()
    out = _base("dino", requested)
    if requested == "stub":
        out["available"] = True
        return out
    try:
        importlib.import_module("shapesplat.frontend.dinov3_real")
    except Exception as exc:
        out["errors"].append(f"RealDINOv3Wrapper import failed: {exc}")
    if not fcfg.get("dino_model_name") and not fcfg.get("dino_checkpoint"):
        out["errors"].append("dino_model_name or dino_checkpoint is not configured")
    elif fcfg.get("dino_checkpoint"):
        ok, err = _checkpoint_ok(fcfg.get("dino_checkpoint"))
        if not ok:
            out["errors"].append(err)
    out["available"] = not out["errors"]
    if requested == "auto" and not out["available"]:
        out["will_fallback"] = True
        out["fallback_target"] = "stub"
        out["warnings"].append("DINO auto backend will fallback to stub")
    return out


def detect_depth_capability(cfg: dict) -> dict:
    """检测 depth backend；仅做配置和 wrapper 可见性检查。"""

    fcfg = cfg.get("frontend", {})
    requested = str(fcfg.get("depth_backend", "stub")).lower()
    out = _base("depth", requested)
    if requested == "stub":
        out["available"] = True
        return out
    try:
        importlib.import_module("shapesplat.frontend.depth_real")
    except Exception as exc:
        out["errors"].append(f"RealDepthWrapper import failed: {exc}")
    if not fcfg.get("depth_model_name") and not fcfg.get("depth_checkpoint"):
        out["errors"].append("depth_model_name or depth_checkpoint is not configured")
    elif fcfg.get("depth_checkpoint"):
        ok, err = _checkpoint_ok(fcfg.get("depth_checkpoint"))
        if not ok:
            out["errors"].append(err)
    out["available"] = not out["errors"]
    if requested == "auto" and not out["available"]:
        out["will_fallback"] = True
        out["fallback_target"] = "stub"
        out["warnings"].append("Depth auto backend will fallback to stub")
    return out


def detect_shape_bank_capability(cfg: dict) -> dict:
    """检测 shape bank；file/auto 只检查 root 是否存在。"""

    scfg = cfg.get("shape_bank", {})
    requested = str(scfg.get("backend", "toy")).lower()
    out = _base("shape_bank", requested)
    if requested == "toy":
        out["available"] = True
        return out
    root = scfg.get("root")
    if root and Path(root).exists():
        out["available"] = True
        out["details"]["root"] = str(root)
    else:
        out["errors"].append("shape_bank.root is missing or does not exist")
    if requested == "auto" and not out["available"] and scfg.get("fallback_to_toy", True):
        out["will_fallback"] = True
        out["fallback_target"] = "toy"
        out["warnings"].append("ShapeBank auto backend will fallback to toy")
    return out


def detect_renderer_capability(cfg: dict) -> dict:
    """检测 renderer backend；real 只检查 module/class 能否 import。"""

    rcfg = cfg.get("renderer", {})
    requested = str(rcfg.get("backend", "soft")).lower()
    out = _base("renderer", requested)
    real_cfg = rcfg.get("real_3dgs", {})
    out["details"].update(
        {
            "backend": requested,
            "library": real_cfg.get("library", "auto"),
            "use_native_contributions": bool(real_cfg.get("use_native_contributions", False)),
            "object_contribution_mode": real_cfg.get("object_contribution_mode", "object_wise_alpha"),
        }
    )
    if requested == "soft":
        out["available"] = True
        return out
    module_name = rcfg.get("real_renderer_module")
    class_name = rcfg.get("real_renderer_class")
    if module_name and class_name:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            out["available"] = True
            out["details"]["library"] = "custom_module"
        except Exception as exc:
            out["errors"].append(f"real renderer import failed: {exc}")
    else:
        try:
            from shapesplat.renderer.real_3dgs_adapter import Real3DGSRendererAdapter

            adapter = Real3DGSRendererAdapter(camera=None, cfg=cfg)
            out["available"] = bool(adapter.available)
            out["details"]["library"] = adapter.library_name or real_cfg.get("library", "auto")
            if not adapter.available:
                out["errors"].append(adapter.error_message or "real 3DGS renderer unavailable")
        except Exception as exc:
            out["errors"].append(f"Real3DGSRendererAdapter detection failed: {exc}")
    if requested == "auto" and not out["available"] and rcfg.get("fallback_to_soft", True):
        out["will_fallback"] = True
        out["fallback_target"] = "soft"
        out["warnings"].append("Renderer auto backend will fallback to soft")
    return out


def detect_external_baseline_capability(cfg: dict, external_cfg: dict | None = None) -> dict:
    """检测 external baseline 配置；不会执行外部命令。"""

    entries = (external_cfg or {}).get("external_baselines", []) if external_cfg else []
    rows = []
    for entry in entries:
        adapter = entry.get("adapter")
        row = _base(entry.get("name", adapter or "external"), str(adapter))
        if adapter == "dummy_external":
            row["available"] = True
        elif adapter == "command_template":
            row["available"] = bool(entry.get("command"))
            if not row["available"]:
                row["warnings"].append("command_template has no command")
        else:
            row["warnings"].append("unknown or disabled external adapter")
        row["details"]["enabled"] = bool(entry.get("enabled", False))
        rows.append(row)
    return {"backend": "external_baselines", "entries": rows, "available": all(r["available"] for r in rows) if rows else True}


def detect_all_capabilities(cfg: dict, external_cfg: dict | None = None) -> dict:
    """统一检测所有 backend capability。

    capability detection 用于本地真实组件接入前快速检查；真实依赖缺失
    时只记录 warning/fallback，不影响默认 stub pipeline。
    """

    caps = {
        "sam": detect_sam_capability(cfg),
        "dino": detect_dino_capability(cfg),
        "depth": detect_depth_capability(cfg),
        "shape_bank": detect_shape_bank_capability(cfg),
        "renderer": detect_renderer_capability(cfg),
        "external_baselines": detect_external_baseline_capability(cfg, external_cfg),
    }
    warnings = []
    for key in ["sam", "dino", "depth", "shape_bank", "renderer"]:
        warnings.extend(caps[key].get("warnings", []))
    caps["overall"] = {
        "ready_for_stub_pipeline": True,
        "ready_for_real_frontend": all(caps[k]["available"] for k in ["sam", "dino", "depth"]),
        "ready_for_real_renderer": caps["renderer"]["available"] and not caps["renderer"].get("will_fallback", False),
        "warnings": warnings,
    }
    return caps
