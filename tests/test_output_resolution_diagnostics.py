from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from shapesplat.experiments.co3dv2_highres_report import collect_resolution_rows, generate_co3dv2_highres_report


def _work_dir(name: str) -> Path:
    root = Path("outputs") / "test_output_resolution_diagnostics" / f"{name}_{uuid.uuid4().hex}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_fake_diag(root: Path, image_id: str = "co3d_000") -> Path:
    sample = root / "per_image" / image_id
    sample.mkdir(parents=True, exist_ok=True)
    diag = {
        "original_image_shape": [3, 479, 640],
        "original_mask_shape": [479, 640],
        "working_image_shape": [3, 479, 640],
        "working_mask_shape": [479, 640],
        "renderer_image_shape": [3, 479, 640],
        "frontend_cache_used": True,
        "frontend_cache_dir": "cache/co3d_000",
        "mask_source": "file",
        "mask_resize_applied": False,
        "mask_resize_mode": "nearest",
        "dino_input_size": 448,
        "dino_feature_shape": [384, 479, 640],
        "debug_iteration_cap_applied": False,
        "visible_steps": 50,
        "hidden_steps": 50,
        "joint_steps": 100,
        "edit_steps": 20,
        "shape_bank_backend": "toy",
        "renderer_backend": "soft",
        "renderer_fallback": None,
        "shape_bank_fallback": True,
    }
    (sample / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    return sample


def test_collect_resolution_rows():
    tmp_path = _work_dir("collect")
    _write_fake_diag(tmp_path)
    rows = collect_resolution_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["working_image_shape"] == [3, 479, 640]
    assert rows[0]["debug_iteration_cap_applied"] is False


def test_generate_co3dv2_highres_report_fake():
    tmp_path = _work_dir("report")
    _write_fake_diag(tmp_path)
    manifest = generate_co3dv2_highres_report(tmp_path, tmp_path / "report")
    assert Path(manifest["report_path"]).exists()
    assert (tmp_path / "report" / "tables" / "resolution_summary.csv").exists()
    text = Path(manifest["report_path"]).read_text(encoding="utf-8")
    assert "single foreground" in text
    assert "thumbnails" in text
