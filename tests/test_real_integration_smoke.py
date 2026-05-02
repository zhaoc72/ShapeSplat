from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from shapesplat.config import load_config
from shapesplat.integration.capabilities import detect_all_capabilities
from shapesplat.integration.config_templates import create_local_backend_template
from shapesplat.integration.report import make_capability_markdown
from shapesplat.integration.smoke import run_real_integration_smoke

ROOT = Path(__file__).resolve().parents[1]


def _tmp_dir(name: str) -> Path:
    path = ROOT / "outputs" / "test_real_integration_smoke_tmp" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_detect_capabilities_stub() -> None:
    cfg = load_config("configs/minimal.yaml")
    caps = detect_all_capabilities(cfg)
    for key in ["sam", "dino", "depth", "renderer"]:
        assert isinstance(caps[key], dict)
    assert caps["overall"]["ready_for_stub_pipeline"] is True


def test_detect_capabilities_auto_fallback() -> None:
    cfg = load_config("configs/local_backend_template.yaml")
    caps = detect_all_capabilities(cfg)
    assert caps["overall"]["ready_for_stub_pipeline"] is True
    assert caps["sam"]["will_fallback"] or caps["dino"]["will_fallback"] or caps["depth"]["will_fallback"]


def test_make_capability_markdown() -> None:
    caps = {
        "sam": {"requested": "auto", "available": False, "will_fallback": True, "fallback_target": "stub", "warnings": ["fallback"]},
        "dino": {"requested": "stub", "available": True, "will_fallback": False, "warnings": []},
        "depth": {"requested": "stub", "available": True, "will_fallback": False, "warnings": []},
        "shape_bank": {"requested": "toy", "available": True, "will_fallback": False, "warnings": []},
        "renderer": {"requested": "soft", "available": True, "will_fallback": False, "warnings": []},
        "external_baselines": {"available": True, "entries": []},
    }
    md = make_capability_markdown(caps)
    assert "Backend" in md
    assert "Requested" in md


def test_run_real_integration_smoke_stub() -> None:
    cfg = load_config("configs/local_backend_template.yaml")
    cfg["image"]["size"] = 32
    cfg["training"]["visible_warmup_iters"] = 1
    cfg["training"]["hidden_prior_iters"] = 1
    cfg["training"]["joint_ownership_iters"] = 1
    cfg["training"]["edit_finetune_iters"] = 1
    out = _tmp_dir("smoke")
    report = run_real_integration_smoke(cfg, None, out, save_cache=True, run_reconstruction=True)
    assert report["status"] in {"success", "success_with_fallback"}
    assert (out / "integration_report.json").exists()
    assert (out / "integration_report.md").exists()
    assert (out / "cache" / "masks.npy").exists()


def test_create_local_backend_template() -> None:
    out = _tmp_dir("template") / "backend.yaml"
    create_local_backend_template(out)
    text = out.read_text(encoding="utf-8")
    assert "sam_backend" in text
    assert "dino_backend" in text
    assert "backend: auto" in text
